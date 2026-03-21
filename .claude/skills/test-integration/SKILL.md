---
description: "Run the automated OpenClaw + MLX Server integration test suite. Manages server lifecycle, runs scenarios, collects logs, and evaluates cache metrics."
---

# Automated Integration Test

## Architecture

The test harness (`test_openclaw_integration.py`) runs externally to OpenClaw and interacts via:
1. **OpenClaw CLI** (`openclaw agent`): Simulates user inputs
2. **Mission Control API** (`http://localhost:3000/api`): Monitors task states
3. **MLX Server Logs** (`logs/`): Evaluates cache metrics

## Running

```bash
python3 test_openclaw_integration.py
```

The `ServerManager` handles:
- Killing zombie `litellm`/`start-llm.py` processes
- Starting MLX server in managed subprocess with log capture
- Polling health endpoint until ready (60s timeout)
- Collecting new session logs into `test_results/<RUN_ID>/`
- Graceful teardown

## Scenarios

### S1: Simple Delegation
User asks Spock to write a hello-world script via dev_Dave.
Expected: Spock creates MC task, spawns sub-agent, task reaches `completed`.

### S2: Multi-Step Pipeline
Research via `res_Archer`, then write via `dev_Dave`.
Expected: Sequential sub-agent handoff, both subtasks complete.

### S3: Stale Task Recovery
Simulates sub-agent timeout. Tests Phase 4 Nudge & Recovery.

### S4: Context Stability Under Load
5 consecutive file read/write operations.
Expected: `cache_match_type` consistently `stable_prefix_shorter`, hit > 90%.

## Evaluation

Results go to `test_results/<RUN_ID>/`:
- `server.log`: Raw server output (scan for `diverge`, `evict`, `flush`, `delete`)
- `cache-session-*.log`: Per-session cache metrics
- Parse with `evaluate_cache_logs()` and `evaluate_server_log()` functions

## Important

- Server prefill can take ~5 minutes for cold start with large models. The harness tolerates this natively (no tight timeout).
- MC polling window is 10 minutes (60 polls * 10s) to account for sub-agent prefill times.
