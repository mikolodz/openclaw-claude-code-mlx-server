# Automated Testing Plan: OpenClaw + MLX Server

This document outlines the synthetic testing strategy to evaluate the OpenClaw architecture against the local MLX LLM server. It focuses on validating task execution, delegation, Mission Control synchronization, sub-agent task briefs, and KV cache performance.

## Test Architecture

The test harness runs externally to OpenClaw (in the `~/mlx` directory) and interacts with the system via three channels:
1. **OpenClaw CLI (`openclaw agent`)**: To simulate user inputs to the Team Leader (`pm_Spock`).
2. **Mission Control API (`http://localhost:3000/api`)**: To monitor task states, agent heartbeats, and goal tree structure.
3. **MLX Server Logs (`~/mlx/logs/`)**: To evaluate cache hit rates, prefix stability, and message integrity.

## Synthetic Test Scenarios

### Scenario 1: End-to-End Simple Delegation (S1)
**Goal:** Verify basic delegation loop, MC task creation, and cache hit stability.
**Action:** User asks Spock to write a Python hello-world script and save it.
**Expected Outcomes:**
- Spock queries `GET /api/projects`.
- Spock creates a goal task and a subtask via `POST /api/tasks` (with full `<task_brief>`).
- Spock spawns `dev_Dave` or `opencode` with the subtask ID.
- Sub-agent marks task `in_progress`, completes the file, then marks task `completed` via Mission Control / proxy.
- Cache hit rates remain > 80% on subsequent turns. No mid-history mutations (FP-2) or volatile field leaks (FP-1) in MLX logs.

### Scenario 2: Multi-Step Delegation Pipeline (S2)
**Goal:** Verify sequential sub-agent handoff and Spock's orchestration capability.
**Action:** User asks Spock to 1) research system stats (via `res_Archer`), and 2) write them to a markdown file (via `dev_Dave`).
**Expected Outcomes:**
- Spock creates 1 Goal, 2 Subtasks.
- Sub-agent 1 completes its subtask and returns result.
- Spock reads result, spawns Sub-agent 2 with the second subtask.
- Both subtasks reach `completed`. Spock reports final success.
- MLX Server cache stability maintained despite multiple `sessions_spawn` tool calls and complex history.

### Scenario 3: Stale Task Recovery (S3)
**Goal:** Verify Phase 4 Nudge & Recovery protocol.
**Action:** User asks Spock a long-running task. The harness artificially intercepts and changes the subtask status to simulate a silent sub-agent timeout, or sets the `updated_at` back 40 minutes in SQLite. Spock is nudged via a heartbeat.
**Expected Outcomes:**
- Spock calls `GET /api/tasks/stale?threshold_minutes=30`.
- Spock identifies the stale task, checks `sessions_history`, and nudges the sub-agent or marks the task `blocked` (session_timeout).
- MLX cache maintains stability during the recovery logic.

### Scenario 4: Context Stability Under Load (S4)
**Goal:** Verify MLX cache prefix reuse during heavy mid-history insertions (OpenClaw `toolResult` format).
**Action:** Perform 5 consecutive file read/write operations via `mlx-test` or `pm_Spock`.
**Expected Outcomes:**
- Tool results are injected cleanly as `toolResult` messages.
- `cache_match_type` consistently shows `stable_prefix_shorter`.
- Cache hit rate stays > 90%.
- No memory leaks / OOM errors in the MLX server.

## Evaluation Methods
- **MC State Validation:** The test script continuously polls the Mission Control API (`/api/summary`) and asserts expected transitions (`pending -> in_progress -> completed`).
- **Brief Validation:** The test script extracts the `details` field from the MC API to verify `<task_brief>` XML formatting.
- **Cache Metric Analysis:** Post-session, the script parses the isolated `test_results/<RUN_ID>/` directory for the specific `session_id`, verifying `stable_prefix_msg_count`, `hit%`, and `cache_selection_source`.
- **Server Debug Analysis:** The script scans the isolated `server.log` for crucial keywords like `diverge`, `evict`, `flush`, and `delete` to detect structural cache failures or OOM mitigations from the local MLX server.

## Automation Script (`test_openclaw_integration.py`)
A Python-based orchestrator utilizing a robust `ServerManager` that:
1. **Server Lifecycle Management:** Hunts down zombie `litellm` processes, starts the MLX server in a managed subprocess, and polls the server health endpoint until it is ready.
2. **Execution:** Uses `subprocess` to trigger `openclaw agent --agent pm_spock ...` commands, inherently tolerating 5+ minute MLX cold prefill times natively without timing out.
3. **Observation:** Uses `requests` to poll `localhost:3000/api` for up to 10 minutes (to account for background sub-agent prefill times), watching for expected task status transitions.
4. **Log Isolation & Teardown:** Gracefully shuts down the LLM server and copies ONLY the newly generated `cache-session-*.log` files and the raw `server.log` into an isolated `test_results/<RUN_ID>/` artifact folder.
5. **Reporting:** Parses the isolated logs to generate a final report on MC task completion, KV cache metrics, and cache divergence anomalies.
