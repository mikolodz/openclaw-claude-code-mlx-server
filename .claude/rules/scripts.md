---
paths:
  - "scripts/**"
  - "test_openclaw_integration.py"
---

# Analysis Scripts & Test Harness Rules

## Scripts

| Script | Purpose | Usage |
|---|---|---|
| `scripts/diff_turns.py` | Per-turn structural diff of server log entries | `python3 scripts/diff_turns.py logs/YYYY-MM-DD/LOGFILE.log [turn_a] [turn_b]` |
| `scripts/inspect_mlx_cache.py` | MLX KVCache API inspection (mock surgery tests) | `python3 scripts/inspect_mlx_cache.py [model_path]` |
| `scripts/probe_session.py` | Synthetic cache validation (4 scenarios) | `python3 scripts/probe_session.py` (requires running server) |

## Log Format

Server logs live in `logs/YYYY-MM-DD/cache-session-HASH.log`. Each entry:
```
[TIMESTAMP] request_id=XXXX direction=prompt|response
{JSON payload}
```

Key fields in `request_meta`: `prompt_tokens`, `matched_prefix_len`, `cache_match_type`, `cache_selection_source`, `stable_prefix_msg_count`, `stable_prefix_token_len`, `session_id`.

## Probe Scenarios

| Scenario | What it tests | Expected result |
|---|---|---|
| `probe-normal` | Pure tail-append | `shorter` cache hit, ~94% |
| `probe-drift` | Trailing whitespace (FP-1) | `stable_prefix_shorter`, normalisation absorbs drift |
| `probe-semantic` | Real content change | `miss`, normalisation does NOT absorb real changes |
| `probe-insert` | Mid-history message insertion (FP-2) | 60% hit via global, prefix preserved |

## Test Harness (`test_openclaw_integration.py`)

Manages server lifecycle (start, wait for ready, teardown), runs OpenClaw CLI commands, polls Mission Control API, collects logs into `test_results/<RUN_ID>/`.
