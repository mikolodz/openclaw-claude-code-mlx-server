# Active Tasks

## Current Prioritised Portions (2026-04-24)

Grouped into portions per the CLAUDE.md Change Protocol (portion = items sharing
root cause or test surface).  Tackle in order.  Each portion = one spec, one
implementation, one verification pass (PI agent via `scripts/pi_like_probe.py`).

| # | Portion | Items | Severity | Why grouped |
|---|---------|-------|----------|-------------|
| ~~**P1**~~ | ~~**Image-identity correctness**~~ | ~~G1 + 1a~~ | ~~P0 correctness~~ | **CLOSED 2026-04-24.** Canonical cache key now carries 2-token image-identity markers; retreat evicts the offending entry.  Verified: text-only regression T3=97.6% (≥ baseline); image intra-session 95.4%/95.2%; cross-session isolation B T1 = 0/589 clean miss (was silently hitting A's KV pre-fix); zero `vlm_retreat` events.  See LESSONS.md "Image Identity Must Be Injected…". |
| ~~**P2**~~ | ~~**Session-turn-store guard**~~ | ~~G3~~ | ~~P0 hit-rate~~ | **CLOSED 2026-04-24.** Write guard and `_message_diff` now compare `_canonicalize_messages` output; `_SessionTurnRecord` stores canonical_messages.  Verified: text regression T3=97.5% (on baseline); image intra-session T2=94.8%; drift-injecting probe with per-turn `message_id` mutation shows `stable_prefix_msg_count=[0, 2, 4]` (was [0, 0, 0] pre-G3).  See LESSONS.md "Session-Turn-Store Guard Must Compare Canonical Form". |
| ~~**P3**~~ | ~~**Generation watchdog**~~ | ~~G2~~ | ~~P0 availability~~ | **CLOSED 2026-04-24.** `GENERATION_WATCHDOG_SECONDS` (default 600) arms a `threading.Timer` after `model_lock.acquire()`; yield loops check the abort flag after each token; on trip all side-effect writes (cache, session-turn, healing) are skipped and the response terminates with `finish_reason="length"`.  Timer cancelled in `finally` before `model_lock.release()`.  Verified: 3s-watchdog forced abort at 3.31s with 111 partial tokens + clean terminal chunk; follow-up request 0.5s later served in 0.53s (lock released).  See LESSONS.md "Generation Must Have A Wall-Clock Deadline". |
| ~~**P4**~~ | ~~**Observability for cache correctness**~~ | ~~3 counters~~ | ~~P1 informational~~ | **CLOSED 2026-04-24.** `_CACHE_METRICS` + `_bump_metric` + thread-local request id.  Surfaces via terminal `🧪` one-liners and per-request log fields.  Verified clean-traffic: `vlm_retreat=0`, `exact_key_rejected_by_model_lcp=0` ✓; `hybrid_trim_miss=1` on image-session T1 following a text session — **real signal that gates P5 as worthwhile.**  See LESSONS.md "Cache-Correctness Metrics". |
| **P5** | **Hybrid-VLM prompt-only checkpoint** | G4 (see complexity notes below) | P1 quality | **DEFERRED 2026-04-24** — the simple "deep-copy `prompt_cache` before `stream_generate`" approach doesn't work (see next row for why).  Real implementation needs a dedicated design pass against mlx-vlm internals.  P4's `hybrid_trim_miss=1` per cross-session cold-start is real but modest; in-session T2+ hit rates are already 95% via healing + full-turn cache.  **Do not re-open until the G4 design note below is addressed.** |
| **P5-note** | **G4 complexity constraints (for the next attempt)** | Design facts discovered while scoping P5 | reference | **The earliest reentry point from `stream_generate_vlm` is AFTER prefill AND after the first decode step** (cache already has `prompt + 1 token`).  Snapshot-before-call is useless (cache state is pre-prefill, not the prompt-only state we need).  A "prompt + 1 token" snapshot fails `full_cover` matching on hybrid because `LCP(entry.model_tokens, req_model) = len(prompt) < len(entry.model_tokens) = len(prompt)+1` → trim required → hybrid rejects → miss.  Partial-layer trim (trim KVCache, keep ArraysCache) is incorrect because linear-attention state depends on accumulated history — giving the network a state that has seen MORE tokens than expected produces wrong decode output.  Two viable paths: **(A) manual prefill via direct `model(input_ids, pixel_values=…, cache=cache)` calls**, snapshot, then bootstrap `stream_generate` from the snapshot with a 1-token kickoff — requires reproducing pixel_values/mask/position_ids handling correctly.  **(B) wrap `stream_generate_vlm` to intercept between prefill and first decode** — relies on mlx-vlm internal yield semantics, fragile to upstream changes.  Both are larger than a single-portion budget.  Recommend: open a dedicated design doc next time, read mlx-vlm source first, treat it as a standalone feature not a themed patch. |
| ~~**P6**~~ | ~~**Cache-key/token-math clean-up (themed)**~~ | ~~1c + 1d + 1e~~ | ~~P2 correctness + robustness~~ | **CLOSED 2026-04-24.** `matched_prefix_len` reporting unified across full-cover/exact-key paths (canonical-space, no spurious model-space -1).  `_cull_redundant_prefixes` now additionally requires `new_model_tokens[:len(entry.model_tokens)] == entry.model_tokens` — defense-in-depth against cross-stream canonical aliasing.  `preserve_thinking` kwarg capability-probed at startup (`🧵 preserve_thinking probe: supported/not`); `_chat_template_extras` and `_compute_msg_token_boundaries` only forward when supported.  Verified: text T3=97.5%, image T2=95.6%, zero counter bumps.  See LESSONS.md "Canonical/Model Math Consistency + Template-Kwarg Robustness". |
| ~~**P7**~~ | ~~**DFlash threading note**~~ | ~~1f~~ | ~~P3 future-proofing~~ | **DISSOLVED 2026-04-24.** DFlash integration fully removed — observed negative performance impact on the active model (Qwen3.6 hybrid VLM).  The threading note has no referent.  Removal covered: `mlx_vlm.speculative` import, 5 Settings fields + env reads, `draft_model` module global, startup drafter load (~50 lines), `_stream_generate_unified` DFlash kwargs branch (~15 lines).  Git history preserves full implementation if upstream accept-rate improves and we want to re-introduce. |
| ~~**P8**~~ | ~~**`reasoning_content` echo duplication (FP-3)**~~ | ~~1 item~~ | ~~P0 hit-rate~~ | **CLOSED 2026-04-24.** After the `_ReasoningStreamSplitter` started emitting `delta.reasoning_content` separately from `delta.content`, pi (and any client that accumulates both channels) echoes turn-N+1 back with `content = thinking+</think>+answer` AND `reasoning_content = thinking`.  Qwen3.6's `preserve_thinking` template duplicates the thinking (once from `reasoning_content`, once from `content`), shifting every subsequent token.  LCP stops 2 tokens short of full-cover on turn 1's stored entry; hybrid cache can't trim → every multi-turn pi session missed 100%.  **Fix**: `_heal_messages` drops `reasoning_content` when `content` already contains `</think>`.  Template extracts thinking from content cleanly in that case (verified: produces byte-identical tokens to turn 1's original generation, LCP = 480/480 = full_cover).  Verified: real 3-turn pi test pre-fix 0% / 0% / 0%, post-fix 0% / 96.3% / 96.6%.  Zero `hybrid_trim_miss` bumps post-fix.  See LESSONS.md "`preserve_thinking` + Echoed `reasoning_content` Is A Cache Assassin". |

**Already tracked in Open Items below, unchanged by this plan:**
- Phase 6/8 regression tests, `probe_session.py` validation, OpenClaw Round 2, SESSION_TURN_STORE-for-VLM, image-swap detection (subsumed by P1/G1).

## Recently Closed (2026-04-24 — VLM cache regression fix)

Qwen3.6 (Qwen3_5_MoeForConditionalGeneration) cache path was broken on pi /
OpenClaw tool-call turns with images.  Fixed in-session:

- **HTTP keep-alive deadlock** — single-threaded HTTPServer parked on LiteLLM's
  keep-alive socket; every other connection queued indefinitely.  Fix: force
  `close_connection=True` in do_POST/do_GET and send `Connection: close` on
  streaming responses so the accept loop stays fair.
- **`_kv_cache_offset` read layer 0** — Qwen3.6 hybrid cache is
  `[ArraysCache, KVCache, ArraysCache, ...]`; ArraysCache has no `.offset`
  so layer 0 returned None and the caller fell back to the canonical
  `matched_prefix_len`, slicing `model_tokens` mid image-pad expansion.
  Fix: scan for the first attention layer that exposes an int `.offset`.
- **Mid-image prefill crash** — mlx-vlm raises
  `Image features and image tokens do not match: tokens N, features M` when
  input_ids contains only a partial span of image-pad ids.  Root cause: the
  cache was trimmed in canonical-token space, but KV depth is in model-token
  space; any canonical trim straddling an image block left partial pads in
  `rest_tokens`.  Fix: store `model_tokens` per cache entry, compute KV trim
  via model-space LCP, and retreat (full prefill) if the LCP still lands
  mid-image as a defence-in-depth.
- **Cross-session KV leakage** — a text-only request with the same system
  prefix as an earlier image session was matching (via canonical block hash)
  an entry whose KV depth was 3× larger than the request's model tokens, so
  the model decoded with stale image attention in scope.  Fix: model-space
  LCP + full-cover preference rejects these cache candidates and falls back
  to a clean miss.
- **Hybrid-cache trimmability** — `can_trim_prompt_cache` returns False for
  Qwen3.6 hybrid.  Any "shorter" match requiring trim would previously
  silently skip the trim and continue with a too-deep KV.  Fix: refuse
  cache reuse when trim is needed but infeasible; prefer candidates that
  are a strict prefix of the request (no trim required).
- **Healing hash canonicalisation** — clients (pi) echo tool_calls with
  `function.arguments` as a native dict; the server emits it as a JSON
  string.  `_get_healing_hash` hashed both forms differently, so pi's
  replay-turn tool_calls never matched the stored healing entry and the
  model saw stripped assistant content, which reshuffled downstream tokens
  and missed the cache.  Fix: normalise `arguments` (string → parsed
  mapping) before hashing.
- **`use_vision` correctness** — `_stream_generate_unified` detected
  "image tokens present" via the old Qwen2_VL token ids and a
  `len(rest_tokens) > 100` fallback; after a correct KV offset the rest
  tokens have zero image pads yet the heuristic still forced pixel_values,
  producing the feature/tokens mismatch.  Fix: resolve `image_token_id`
  from the model config at load time and check rest_tokens precisely.

Result: pi tool-call sessions hit 96.9%-99.0% on turns ≥ 2 (text-only and
image paths).  Real pi E2E verified.  Cross-session contamination
eliminated.

## Open Items

### Next-step bugs (tracked as GOALS.md G1–G3)

- [ ] **G1 — Image-identity cache correctness.**  image_pad_token_id is a
      single id repeated 3120× per image; LCP can't distinguish image A
      from image B at the same canonical position.  Hash image payloads
      into the cache key so content change is visible at the token
      stream.
- [ ] **G2 — Generation watchdog + model_lock recovery.**  No per-request
      deadline exists; a hung generation freezes the server.  Add wall-
      clock abort that releases model_lock and evicts partial cache.
- [ ] **G3 — Canonical-form write guard for SESSION_TURN_STORE.**  The
      guard's whitespace-only normalisation trips on OpenClaw's per-turn
      message_id / Inbound Context / runtime drift and freezes M3
      recovery.  Switch the guard to `_canonicalize_messages` output.

### Cache quality (tracked as GOALS.md G4)

- [ ] **G4 — Pre-generation KV checkpoint for hybrid VLMs.**  Deep-copy
      `prompt_cache` before `stream_generate` extends it, insert as the
      prompt-only entry.  Removes the `can_trim_prompt_cache` gate that
      hybrid VLM always fails.

### Validation (from Phase 6/8)

- [ ] **Regression test Phase 6 Step 5**: Run `_canonicalize_messages()` on the raw system message from `logs/2026-03-14/cache-session-bd7087f6ef7a0c43.log`. Assert all 8 workspace files survive intact in original. Assert exactly 1 `__STABLE_INBOUND_META__` in canonical form.
- [ ] **Probe session validation (Phase 6)**: Run `scripts/probe_session.py` all scenarios. Hit rates must match or exceed Phase 3 baseline. _Note 2026-04-24: scenario "normal" now misses intentionally because the script forces an assistant message whose tokens do not match the model's T1 generation — with strict model-space cache correctness, that legitimately diverges.  Pi-style agentic flows (which healing-round-trip the raw generation) are covered by the new `scripts/pi_like_probe.py`._
- [ ] **Regression test Phase 8 fixes**: Run `scripts/probe_session.py` all scenarios. Confirm baseline. Verify Fix B idempotency on T2 (block present) vs T7 (block absent) system messages.
- [x] **VLM canonical pipeline validation**: Start server with VLM model, run 2-turn session, confirm `cache_key_delta_chars > 0` and ~98% hit on turn 2. (2026-04-24: confirmed 95.6–98.5% hits across 5-turn image+tool sessions.)
- [ ] **SESSION_TURN_STORE for VLM via canonical messages**: Currently uses original messages for VLM. Global cache hit is fixed by Phase 8 Fix C and now additionally guarded by model-space LCP; M3 secondary lookup remains disabled for VLM sessions with system-message drift.
- [ ] **Same-position image change detection**: image_pad_token_id is constant, so LCP cannot distinguish "image A" KV from "image B" text-identical request.  If a user swaps images mid-session, current code reuses stale image KV.  Needs hashing image bytes into the canonical stream for full safety.

### OpenClaw Testing (blocked)

- [ ] **OpenClaw Round 2 -- working tool calls required**. Blocked on model tool-calling capability. Three options:
  - Option A: Add tool-call extraction layer to MLX server
  - Option B: Use GLM-4.7-Flash via ollama (known to tool-call)
  - Option C: Inspect existing `pm_Spock` + GLM-4.7-Flash cron session logs
- [ ] **Context compaction test**: Run 20+ turn OpenClaw session to trigger `contextPruning.mode: cache-ttl`. Observe whether stable-prefix layer recovers correctly from shortened context.

### Open Questions (low priority)

- Can LiteLLM proxy usefully pre-normalise before MLX sees the request? (Probably not needed -- M4 at MLX level is sufficient.)
- Are there unused MLX cache primitives (disk serialisation, block pinning)?
