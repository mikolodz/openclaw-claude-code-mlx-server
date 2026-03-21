# Active Tasks

## Open Items

### Validation (from Phase 6/8)

- [ ] **Regression test Phase 6 Step 5**: Run `_canonicalize_messages()` on the raw system message from `logs/2026-03-14/cache-session-bd7087f6ef7a0c43.log`. Assert all 8 workspace files survive intact in original. Assert exactly 1 `__STABLE_INBOUND_META__` in canonical form.
- [ ] **Probe session validation (Phase 6)**: Run `scripts/probe_session.py` all scenarios. Hit rates must match or exceed Phase 3 baseline.
- [ ] **Regression test Phase 8 fixes**: Run `scripts/probe_session.py` all scenarios. Confirm baseline. Verify Fix B idempotency on T2 (block present) vs T7 (block absent) system messages.
- [ ] **VLM canonical pipeline validation**: Start server with VLM model, run 2-turn session, confirm `cache_key_delta_chars > 0` and ~98% hit on turn 2.
- [ ] **SESSION_TURN_STORE for VLM via canonical messages**: Currently uses original messages for VLM. Global cache hit is fixed by Phase 8 Fix C, but M3 secondary lookup remains disabled for VLM sessions with system-message drift.

### OpenClaw Testing (blocked)

- [ ] **OpenClaw Round 2 -- working tool calls required**. Blocked on model tool-calling capability. Three options:
  - Option A: Add tool-call extraction layer to MLX server
  - Option B: Use GLM-4.7-Flash via ollama (known to tool-call)
  - Option C: Inspect existing `pm_Spock` + GLM-4.7-Flash cron session logs
- [ ] **Context compaction test**: Run 20+ turn OpenClaw session to trigger `contextPruning.mode: cache-ttl`. Observe whether stable-prefix layer recovers correctly from shortened context.

### Open Questions (low priority)

- Can LiteLLM proxy usefully pre-normalise before MLX sees the request? (Probably not needed -- M4 at MLX level is sufficient.)
- Are there unused MLX cache primitives (disk serialisation, block pinning)?
