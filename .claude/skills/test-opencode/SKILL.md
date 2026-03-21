---
description: "Run the OpenCode real-traffic cache test protocol against the local MLX server. Use when validating cache behavior with OpenCode client sessions."
---

# OpenCode Cache Test Protocol

## Prerequisites

1. MLX server running: `python3.12 install_and_run.py` then `curl -s http://127.0.0.1:4000/v1/models`
2. OpenCode installed: `opencode --version` and `opencode auth list` (must show `mlx-local` provider at `http://127.0.0.1:4000/v1`)
3. Clean logs: `rm -rf logs/$(date +%Y-%m-%d)/`

## Test Steps

### Turn 1 -- Cold start with tool use
```bash
opencode run --model mlx-local/openai/local \
  "What is the current date and what operating system are we on? Use bash to check both."
```
Capture session ID: `opencode session list --max-count 1 --format json`

### Turn 2 -- Web fetch (external content)
```bash
opencode run --model mlx-local/openai/local --session <SESSION_ID> \
  "Use web_fetch to fetch https://httpbin.org/json and tell me what JSON fields are in the response."
```

### Turn 3 -- Sub-agent spawn
```bash
opencode run --model mlx-local/openai/local --session <SESSION_ID> \
  "Use the Task tool to launch an agent that will fetch https://httpbin.org/json and return a summary."
```

### Turn 4 -- Context recall
```bash
opencode run --model mlx-local/openai/local --session <SESSION_ID> \
  "Based on everything in this session, write a 2-sentence recap of each turn."
```

## Analysis

### Cache metrics per turn
```bash
python3 -c "
import json, re
from pathlib import Path
logf = Path('logs/$(date +%Y-%m-%d)/<LOGFILE>.log')
with open(logf) as f: content = f.read()
pattern = re.compile(r'\[[\d\-T:\.]+\] request_id=(\w+) direction=(\w+)\n(.*?)(?=\n\[[\d\-T:\.]+\] request_id=|\Z)', re.DOTALL)
entries = pattern.findall(content)
prompts = [(rid, body) for rid, d, body in entries if d == 'prompt']
print(f'Turns: {len(prompts)}')
for i, (rid, body) in enumerate(prompts):
    d = json.loads(body); rm = d.get('request_meta', {}); msgs = d.get('messages', [])
    pt = rm.get('prompt_tokens', 0); mpl = rm.get('matched_prefix_len', 0)
    cm = rm.get('cache_match_type', '?'); hit = round(100*mpl/pt, 1) if pt else 0
    spm = rm.get('stable_prefix_msg_count', 0)
    print(f'Turn {i}: msgs={len(msgs)} tokens={pt} cache={cm}({hit}%) sp_msgs={spm}')
"
```

### Message mutation check
```bash
python3 -c "
import json, re, hashlib
from pathlib import Path
logf = Path('logs/$(date +%Y-%m-%d)/<LOGFILE>.log')
with open(logf) as f: content = f.read()
pattern = re.compile(r'\[[\d\-T:\.]+\] request_id=(\w+) direction=(\w+)\n(.*?)(?=\n\[[\d\-T:\.]+\] request_id=|\Z)', re.DOTALL)
prompts = [json.loads(body) for rid, d, body in pattern.findall(content) if d == 'prompt']
max_msgs = max(len(p.get('messages', [])) for p in prompts)
for idx in range(max_msgs):
    hashes = []
    for t, p in enumerate(prompts):
        msgs = p.get('messages', [])
        if idx < len(msgs):
            h = hashlib.sha256(json.dumps(msgs[idx], sort_keys=True).encode()).hexdigest()[:12]
            hashes.append((t, msgs[idx].get('role','?'), h))
    if hashes:
        status = 'STABLE' if all(h == hashes[0][2] for _,_,h in hashes) else '*** MUTATED ***'
        print(f'msg[{idx:2d}] role={hashes[0][1]:9s} turns={[t for t,_,_ in hashes]} {status}')
"
```

### diff_turns.py
```bash
NUM_TURNS=4
for i in $(seq 0 $((NUM_TURNS-2))); do
  j=$((i+1)); echo "=== Turn $i -> Turn $j ==="
  python3 scripts/diff_turns.py logs/$(date +%Y-%m-%d)/<LOGFILE>.log $i $j
done
```

## Expected Healthy Results

| Metric | Expected | Problem if... |
|---|---|---|
| Turn 1 cache_match_type | `miss` | `shorter` = leaked cache |
| Turn 2+ cache_match_type | `shorter` | `miss` = system message mutated |
| Turn 2+ hit% | >= 95% | < 80% = hash chain break |
| Mutation check | All `STABLE` | `MUTATED` = new failure pattern |

## Known: OpenCode strips tool results from history (pure tail-append). Sub-agent runs in-process (no separate HTTP request).
