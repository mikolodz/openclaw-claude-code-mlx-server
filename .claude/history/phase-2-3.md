# Phase 2-3: Implementation & Validation (COMPLETE)

Completed: 2026-03-13

## Phase 2 -- Implementation

All milestones implemented in `start-llm.py`:

- [x] **M1** -- `_message_diff(prev, curr)`: Returns `stable_prefix_count` + change descriptors
- [x] **M2** -- `_stable_prefix_token_len()` + `SESSION_TURN_STORE`: Secondary cache lookup
- [x] **M3** -- Mid-history insertion handling via M2
- [x] **M4** -- `_normalize_message_content_for_diff()`: Strips whitespace for diff key
- [x] **M4 extended** -- Real-traffic audit: no per-message volatile fields in OpenCode sessions
- [x] **M5** -- `_SessionTurnRecord` storage in both streaming/non-streaming paths
- [x] **M6** -- Telemetry: `stable_prefix_msg_count`, `stable_prefix_token_len`, `stable_prefix_diff`

### Key symbols added
- `_SessionTurnRecord` dataclass
- `SESSION_TURN_STORE` (dict, protected by `prompt_cache_lock`)
- `_normalize_message_content_for_diff(msg)`
- `_message_diff(prev, curr)`
- `_stable_prefix_token_len(session_id, curr_msgs)`
- `_compute_msg_token_boundaries(messages, prompt_tokens)`
- `_update_session_turn_store(session_id, messages, prompt_tokens)`

## Phase 3 -- Validation

- [x] Probe harness all 4 scenarios passing
- [x] L1 boundary computation fixed (cumulative `apply_chat_template`)
- [x] Multi-agent write guard added to `SESSION_TURN_STORE`
- [x] Real-traffic OpenCode audit (Round 1): 6 turns, 99.3-99.8% hit, pure tail-append
- [x] Real-traffic OpenCode audit (Round 2): 4 turns with bash/webfetch/Task, 99.2-99.8% hit
- [x] Regression check: secondary only fires when global match is weaker
- [x] OpenClaw Round 1: INVALIDATED (Qwen3.5 didn't emit structured tool_calls)

### Confirmed from real traffic
- OpenCode strips tool results from history (pure tail-append)
- `<think>` blocks survive intact (healing store works)
- Task sub-agent runs in-process (no separate HTTP request)
- System message completely stable in OpenCode sessions
- OpenClaw timestamp `[Day YYYY-MM-DD HH:MM TZ]` is frozen at send time (not volatile)
