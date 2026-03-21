---
description: "Run the OpenClaw real-traffic cache test protocol. Use when validating cache behavior with OpenClaw's tool-result injection and multi-agent patterns."
---

# OpenClaw Cache Test Protocol

## Why OpenClaw (vs OpenCode)

OpenCode strips tool results from history (pure tail-append). OpenClaw keeps `toolResult` messages interleaved in history -- this is the FP-2 (mid-history injection) scenario. Each tool call adds 2 messages: assistant `toolCall` + `toolResult`.

## Prerequisites

1. MLX server running: `python3.12 install_and_run.py`
2. OpenClaw gateway running: `openclaw gateway status`
3. Create `mlx-test` agent:
   ```bash
   openclaw agents add mlx-test \
     --model litellm-local/openai/local \
     --workspace /tmp/openclaw-mlx-test \
     --non-interactive
   ```
4. Clean logs: `rm -rf logs/$(date +%Y-%m-%d)/`

## Test Steps

### Step 0 -- Cold probe
```bash
openclaw agent --agent mlx-test --message "Reply with exactly: PONG" --json
```
Save the `sessionId` from output.

### Step 1 -- Bash tool
```bash
openclaw agent --agent mlx-test --session-id <SID> \
  --message "What is the current date and OS? Use exec tool: uname -a && date" --json
```

### Step 2 -- Web fetch
```bash
openclaw agent --agent mlx-test --session-id <SID> \
  --message "Use web_fetch to fetch https://httpbin.org/json and list the JSON fields." --json
```

### Step 3 -- File write + read
```bash
openclaw agent --agent mlx-test --session-id <SID> \
  --message "Write 'cache_test_marker_turn3' to /tmp/oc_claw_test.txt, then read it back." --json
```

### Step 4 -- Sub-agent spawn
```bash
openclaw agent --agent mlx-test --session-id <SID> \
  --message "Use sessions_spawn to launch a sub-agent: 'Read /tmp/oc_claw_test.txt and return content'. Report what it found." --json
```

### Step 5 -- Context recall
```bash
openclaw agent --agent mlx-test --session-id <SID> \
  --message "Recap what happened in each turn of this session." --json
```

## Key Questions

| # | Question | Hypothesis |
|---|---|---|
| Q1 | Are `toolResult` messages forwarded verbatim? | Yes (seen in JSONL files) |
| Q2 | Are they byte-stable across turns? | Unknown |
| Q3 | Does OpenClaw inject `Current time:`? | Yes, frozen at send time (confirmed Round 1) |
| Q5 | Does `sessions_spawn` create new HTTP request? | Unknown |
| Q7 | Hit rate with real tool use? | >= 90% if FP-1/FP-2 mitigations work |

## Analysis

Use same scripts as OpenCode protocol (cache metrics, mutation check, diff_turns.py). Additionally:

### Tool result message format audit
```bash
python3 -c "
import json, re
from pathlib import Path
logf = Path('logs/$(date +%Y-%m-%d)/<LOGFILE>.log')
with open(logf) as f: content = f.read()
pattern = re.compile(r'\[[\d\-T:\.]+\] request_id=(\w+) direction=(\w+)\n(.*?)(?=\n\[[\d\-T:\.]+\] request_id=|\Z)', re.DOTALL)
prompts = [json.loads(body) for rid, d, body in pattern.findall(content) if d == 'prompt']
for t, p in enumerate(prompts):
    msgs = p.get('messages', [])
    roles = [m.get('role','?') for m in msgs]
    print(f'Turn {t}: {len(msgs)} msgs -- roles: {roles}')
    for i, m in enumerate(msgs):
        if m.get('role') in ('tool', 'toolResult'):
            c = m.get('content','')
            c_text = str(c)[:80] if isinstance(c, str) else ' '.join(str(x.get('text','')) for x in c)[:80]
            print(f'  msg[{i}] role={m[\"role\"]} preview={repr(c_text)}')
"
```

## Expected Results

| Metric | Expected | Problem if... |
|---|---|---|
| Turn 2+ hit% | >= 80% | < 60% = toolResult messages mutating (FP-2) |
| stable_prefix_msg_count | > 0 on turn 2+ | 0 = session_id not passed |
| Mutation check | All STABLE | MUTATED = new failure pattern |
| Sub-agent logs | 1 or 2 files | 3+ = unexpected parallel sessions |

## Known Limitations

- `mlx-test` has no custom system prompt (may differ from production agents)
- Context pruning (5m TTL) may trigger if gaps between turns are long
- Blocked on model tool-calling capability (Qwen3.5 does not emit structured `tool_calls`)
