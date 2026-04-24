# Lessons Learned

Hard-won knowledge from Phases 0-8. Read this before touching cache logic.

## Failure Patterns (confirmed, documented)

| ID | Name | Trigger | Impact | Mitigation |
|---|---|---|---|---|
| FP-1 | Early Drift | Trailing whitespace, timestamp, or `<system-reminder>` change in system message | 0% hit (total miss) -- hash chain breaks at block 0 | `_normalize_message_content_for_diff()` strips whitespace; `_canonicalize_messages()` handles volatile fields |
| FP-2 | Mid-History Insertion | Tool result spliced mid-history (OpenClaw `toolResult` messages) | Partial hit (prefix only) -- hash chain breaks at insertion point | `SESSION_TURN_STORE` + `_stable_prefix_token_len()` recovers prefix KV |
| FP-3 | `reasoning_content` Echo Duplication | Client (pi) echoes assistant message with BOTH `content` = `thinking + </think> + answer` AND `reasoning_content` = thinking. Qwen3.6 template with `preserve_thinking=True` injects `<think>…</think>` from `reasoning_content`, then renders `content` verbatim — thinking appears twice, LCP stops 2 tokens short of full-cover, hybrid cache cannot trim, every turn = cold miss | 0% hit on every multi-turn pi session (confirmed 2026-04-24) | `_heal_messages` drops `reasoning_content` when `content` already contains `</think>`; template's preserve_thinking extracts thinking from content cleanly, rendering matches the model's turn-1 generation byte-for-byte → `full_cover` hit |

## Architecture Lessons

### Dual Pipeline is Non-Negotiable
Phase 5 proved that in-place mutation of the rendered prompt is catastrophic. A `re.DOTALL` regex meant to strip a JSON block consumed 17,835 chars of agent identity (AGENTS.md, SOUL.md, TOOLS.md). The model was blind to its entire operational protocol.

**Rule**: Cache key and model input must be computed from separate message copies. `_canonicalize_messages()` returns `(original, canonical)`. They never recombine.

### `re.DOTALL` on Full Rendered Strings is Forbidden
Phase 5 + Phase 5 follow-up: Two separate bugs from DOTALL patterns. First was too greedy (consumed 17k chars). Second was too strict (never matched, causing duplicate block injection).

**Rule**: Post-render scrubs in `_scrub_cache_key()` must be atomic, line-scoped. Complex multi-section patterns go in `_canonicalize_messages()` at the struct level.

### RoPE Makes Suffix KV Reuse Invalid
MLX `KVCache.state=` surgery is physically possible (confirmed via mock tests). But RoPE position encodings are baked into K vectors. After a mid-history insertion, suffix messages shift positions -- their cached KV has wrong attention scores. Only prefix KV is safe to reuse.

**Rule**: Only tail-trim is used. Candidate B (segment surgery) is deferred indefinitely.

### Token Boundary Mismatch (Phase 7 Fix 1)
When canonical and original pipelines differ in token count (e.g., 200-token JSON -> 4-token `__STABLE_INBOUND_META__`), using `matched_prefix_len` (canonical count) to slice `model_tokens` (original sequence) produces wrong RoPE positions.

**Rule**: Always use `_kv_cache_offset(cache)` to read the actual KV depth from `cache[0].offset`. Never use canonical token counts as indices into original token arrays.

### Inbound Context Block Can Be Present or Absent
OpenClaw dynamically includes/omits the `## Group Chat Context / ## Inbound Context` block per turn. Phase 8 confirmed: T2 had 44,059 chars, T7 had 43,478 chars. The 581-char difference broke 7,534 tokens of cache.

**Rule**: `_canonicalize_inbound_context_block()` must produce identical output regardless of block presence/absence. Both cases yield `__STABLE_INBOUND_CONTEXT_SECTION__\n# Project Context...`.

### `PROMPT_CACHE_SESSION_MAX_IDLE_SECONDS=0` Means Infinite
Setting TTL to 0 caused all SESSION_TURN_STORE records to be pruned on every write (0s is always expired). Phase 8 Fix A: 0 = "never expire".

### Longest-Candidate Selection is Critical
Phase 4 Fix 4: When a short heartbeat entry and a long conversation entry share the same block hash (identical prefix), the old code picked arbitrarily (set iteration order). Result: 2.6% hit instead of 93%. Fix: always pick deepest matching candidate at each block level.

### `_cull_redundant_prefixes` Can Evict Useful Entries
When a longer cache entry is inserted, `_cull_redundant_prefixes` deletes shorter entries that are strict prefixes. This can remove T1's cache before T2's stable-prefix lookup can use it. Pre-existing behavior -- do not fix without real-session evidence.

### OOM From Unbounded Cache (Phase 4)
`PROMPT_CACHE_MAX_ENTRIES_GLOBAL=64` at 20k tokens * 2.6 GB/entry = 167 GB theoretical. Also: `_insert_cache_entries` did unconditional `deepcopy` of full KV tensor on every turn. Fixed: max entries = 16, deepcopy gated to tool-call turns only, `mx.metal.clear_cache()` on eviction.

### Canonical Cache Key Length ≠ KV Depth For VLM (Phase 9, 2026-04-24)
`LRUPromptCache` used to key entries by `canonical_tokens` only.  For VLM models that expand a single `<|image_pad|>` canonical token into N image features (Qwen3.6: 3120 tokens for a 1668×1912 image), canonical tokens and the KV state are in different spaces.  Any trim computed as `len(cached_canonical) - matched_canonical_prefix_len` is meaningless for the KV, and canonical block hashes collide across requests with different image-expansion shapes, producing catastrophic KV leakage.

**Rule**: Cache entries must store BOTH the canonical key and the `model_tokens` sequence that actually built the KV state.  Candidate selection starts from canonical block hashes (fast), then validates and trims via model-space LCP (correct).  A single `_kv_cache_offset(cache)` read after trim is the safe way for the caller to know where to start prefilling; canonical matched-prefix counts must never be used to slice `model_tokens`.

### Hybrid VLM Caches Are Not Trimmable (Phase 9, 2026-04-24)
Qwen3.6 (Qwen3_5MoeForConditionalGeneration) builds its prompt cache from a mix of `ArraysCache` (linear-attention / mamba state, no position concept) and `KVCache` (regular attention).  `can_trim_prompt_cache` returns False because `ArraysCache.is_trimmable()` inherits the `_BaseCache` default (False).  Any "shorter" partial cache hit that requires trimming is therefore impossible on this model family.

**Rule**: Cache reuse for hybrid VLMs works only when the new request's `model_tokens` is a strict superset-by-prefix of a stored entry's `model_tokens` (no trim needed).  The block-hash candidate ranker must prefer such "full-cover" entries over longer candidates that would require infeasible trims; the fallback is a clean miss rather than a silently-corrupt reuse.

### Single-Threaded HTTPServer + HTTP/1.1 Keep-Alive Deadlocks (Phase 9, 2026-04-24)
mlx-vlm 0.4.4's `generate_step` calls `mx.async_eval` outside a stream context, which crashes in worker threads — so the server must serve single-threaded.  `BaseHTTPRequestHandler` defaults to HTTP/1.1 keep-alive: after one request completes, `handle_one_request` loops waiting for the next request on the same socket.  LiteLLM's connection pool opens multiple long-lived sockets; the server gets parked on whichever socket it first accepted and every other client waits forever.

**Rule**: With `HTTPServer` (non-Threading) the handler must force `self.close_connection = True` in `do_POST`/`do_GET` and emit `Connection: close` on streaming responses.  That keeps the single-threaded accept loop fair without reverting to the threading model that mlx-vlm's stream code forbids.

### `preserve_thinking` + Echoed `reasoning_content` Is A Cache Assassin (FP-3, 2026-04-24)
After the streaming reasoning splitter (`_ReasoningStreamSplitter`) started emitting `delta.reasoning_content` separately from `delta.content`, pi (and any other client that accumulates both channels) echoes the assistant turn back with BOTH fields populated:
- `content = thinking_text + "</think>\n\n" + answer`  (pi flattens so non-reasoning-aware APIs still see coherent text)
- `reasoning_content = thinking_text`

Qwen3.6's `chat_template.jinja` with `preserve_thinking=True` interprets `reasoning_content` by wrapping it in `<think>…</think>` at render time — but then *also* renders the content field verbatim, which already contains the thinking + `</think>` + answer.  Result: the thinking block appears twice with two `</think>` markers, shifting every downstream token; the new prompt's LCP against turn 1's stored entry stops exactly 2 tokens short of full cover (111/113 gen tokens match, then the second thinking kicks in).  Qwen3.6's hybrid cache cannot trim, so the candidate is rejected and every follow-up turn is a cold miss (0% hit on real 3-turn pi sessions, confirmed via `logs/2026-04-24/cache-session-4ccf4d213d201ba5.log`).

**Rule**: In `_heal_messages`, drop `reasoning_content` from any assistant message whose `content` already contains `</think>`.  The Qwen3.6 template's preserve_thinking pass extracts the thinking block from content correctly in that case (verified: produces byte-identical tokens to the original turn-1 generation), so no context is lost.  Keep `reasoning_content` through when content is stripped — that is the legitimate supplementary case for clients that DID receive the two channels and store them cleanly.  The rule is one-way: never synthesise `reasoning_content` server-side, only remove the redundant copy.

**Regression signal**: on multi-turn pi/OpenClaw traffic with `enable_thinking=True`, the T2 terminal line drops to `🔴 Cache: 0.0% (miss)` and `hybrid_trim_miss` bumps twice per request (primary lookup + stable-prefix fallback).  If you see that pattern, re-diff `logs/<date>/cache-session-*.log` turn 0 vs turn 1 and inspect the assistant message: if `content` has `</think>` AND `reasoning_content` is set, this rule has regressed.

### Healing Hash Must Canonicalise Tool-Call Arguments (Phase 9, 2026-04-24)
OpenAI spec emits `function.arguments` as a JSON string.  Real clients (pi) round-trip them back as native dicts.  If the healing hash uses the raw `tool_calls` list verbatim, the server's stored hash (string form) never matches the client's replay hash (dict form), so the assistant's content never gets re-healed and every tool-call turn falls off the cache.

**Rule**: Before hashing, normalise each tool call's `function.arguments` from string → parsed mapping (or a `{"__raw__": str}` placeholder when parsing fails).  This makes the hash stable across any client/server serialisation choice.

### Image Identity Must Be Injected Into The Canonical Cache Key (Portion 1 / G1, 2026-04-24)
The VLM chat template renders a single `<|image_pad|>` token per image in the canonical text (the 3120× expansion happens downstream in `vlm_prepare_inputs`, on the model path only).  Result: two requests with the same message structure but DIFFERENT image payloads produce byte-identical canonical token streams.  The block-hash index and model-space LCP both accept image-A's cached KV for image-B's request — the model silently decodes using the wrong image attention.  Partial-image retreat cannot save this case because the cut doesn't land mid-pad; it lands cleanly AFTER a full (but wrong) image span.  Verified: pre-G1 session B T1 with a different image at the same canonical position would have hit 589/589 against session A's KV; post-G1 it is 0/589.

**Rule**: For VLM requests with images, compute a stable SHA-256 per image and inject a 2-token marker (48 bits of entropy, reserved id range `0x7F000000..0x7FFFFFFF` — above any real vocab, still uint32-encodable for the block-hash chain) immediately after each `<|image_pad|>` position in the canonical token stream.  Markers live in canonical space ONLY — they never touch `model_tokens` or the model's view, preserving the dual-pipeline invariant.  Env flag `CACHE_VLM_IMAGE_IDENTITY` (default on) governs the behaviour.

**Corollary — Retreat Must Evict (Portion 1 / 1a)**: when `_vlm_cache_covers_partial_image` fires and we null the local `prompt_cache` to force a fresh prefill, the offending entry must also be removed from `PROMPT_CACHE` (`PROMPT_CACHE.delete(model, cache_session_tokens)`).  Otherwise it keeps winning candidate selection next turn and the retreat becomes a starvation loop.  With G1 active the retreat becomes a defense-in-depth line that should not trigger on clean traffic — verified: zero `vlm_retreat` events in the Portion 1 probe runs.

### Session-Turn-Store Guard Must Compare Canonical Form (Portion 2 / G3, 2026-04-24)
`SESSION_TURN_STORE`'s write guard, and the `_message_diff` used by `_stable_prefix_token_len`, used `_normalize_message_content_for_diff` — whitespace-only.  OpenClaw drifts per-turn on `message_id`, sub-agent `Stats: runtime`, Inbound Context block variants, timestamps, and `<system-reminder>` injections.  Any single drifted field made the guard reject the write as "prior message changed" → the record froze at the last-clean turn → M3 stable-prefix layer stopped advancing → the session silently fell back to global-only cache hits for the rest of its life.  Verified: pre-G3 a 3-turn session with a drifting `message_id` would show `stable_prefix_msg_count=[0, 0, 0]`; post-G3 it shows `[0, 2, 4]`.

**Rule**: The write guard AND `_message_diff` must compare `_canonicalize_messages` output — the same canonical form the cache key is derived from.  `_SessionTurnRecord` stores `canonical_messages` (not raw messages) because that is the only form ever used for diff/guard.  Consequence/invariant: if two messages produce the same canonical prompt tokens, they produce the same "stable prefix" accounting.  Drift that is cache-key-invisible must also be stable-prefix-invisible.

### Generation Must Have A Wall-Clock Deadline (Portion 3 / G2, 2026-04-24)
`model_lock` held `blocking=True` for the full generation has no wall-clock bound.  A stall (pathological sampler, decode collapse, rare Metal issue) keeps the lock forever; every queued client waits for SIGTERM.  Availability failure mode, not a cache failure mode — but it made the server impossible to share in staging.

**Rule**: After `model_lock.acquire()` start a `threading.Timer(generation_watchdog_seconds, Event.set)`.  Yield loops (streaming + non-streaming) check `Event.is_set()` after each token; on trip, `generation_aborted=True`, break.  On abort, skip every side-effect that depends on clean completion: `_insert_cache_entries` (partial KV with a matching canonical key would corrupt future requests), `_update_session_turn_store` (half-state record misreports stable-prefix boundary), `HEALING_STORE` (partial-text hash would replay truncated content on legitimate client retry).  Terminate the response with `finish_reason="length"` (OpenAI-standard truncation signal).  Cancel the timer in `finally` — idempotent on `threading.Timer`.  Env `GENERATION_WATCHDOG_SECONDS` (default 600 = 10min; `0` disables).  Cooperative abort is chosen over SIGALRM because a signal during a Metal encoder submission can corrupt GPU state and force a model reload — worse than the problem being solved.  Verified: 4096-max-tokens request at 3s watchdog aborted at 3.31s with 419 chars partial content; a follow-up request 0.5s later completed in 0.53s (lock cleanly released).

### Cache-Correctness Metrics (Portion 4, 2026-04-24)
Three P1-P3 invariants are not observable from token throughput alone: "G1 never retreats," "canonical keys are unique per distinct model stream," "hybrid-VLM non-trimmable rejections are rare."  Without counters these are assumed true forever — which is how Phase 9 regressions became silent contamination bugs in the first place.

**Rule**: Instrument three server-lifetime counters behind a lock — `vlm_retreat`, `exact_key_rejected_by_model_lcp`, `hybrid_trim_miss` — and surface them two ways: (a) a terminal `🧪` line per-bump so operators see events immediately, (b) per-request log fields `cache_metrics_snapshot` and `cache_metrics_bumped_this_request` for after-the-fact audit.  Bump sites pass a thread-local request id (set at `do_POST` entry, cleared in `finally`) so internal cache-layer bumps can attribute to the originating request without threading the id through every call.

**Interpretation**:
- `vlm_retreat > 0` **and** `exact_key_rejected_by_model_lcp > 0` are **correctness alarms** — fix root cause immediately.
- `hybrid_trim_miss > 0` is **not an alarm** but a utilization signal.  A text session followed by an image session will bump it once because shared canonical block hashes make the text entry a (rejected) candidate.  Rate > ~1 per image-session T1 would justify G4 (pre-generation prompt-only checkpoint).  Confirmed post-instrumentation: 1 bump on first image-session T1 following a text session — gates P5 as worthwhile.

### Why G4 Is Harder Than "Just Deep-Copy Before Generate" (deferred 2026-04-24)
The master list proposed "deep-copy `prompt_cache` before `stream_generate` extends it with generated tokens; insert that as the prompt-only entry."  Scoping found this does not work:

- `stream_generate_vlm` is a generator.  The earliest reentry point is AFTER prefill AND after the first decode step.  At that yield the cache already holds `prompt + 1 generated token`.
- Snapshot-before-call is useless: cache state at that moment is the pre-prefill state (the partial-hit prefix or empty for a cold miss), not the post-prefill prompt-only state we want.
- A "prompt + 1 token" snapshot fails `full_cover` matching: a future request whose `model_tokens = prompt_model_tokens` gives `LCP = len(prompt) < len(entry.model_tokens) = len(prompt)+1` → trim required → hybrid rejects → miss.  The stored entry would never match anything useful.
- Partial-layer trim (trim KVCache layers, leave ArraysCache untouched) is incorrect: the linear-attention state is an accumulated function of ALL history up to current position — giving the network a state that has "seen" more tokens than the full-attention layers think exist produces wrong decode output (invariant-violating).
- "Lie about depth" (store entry with `model_tokens = prompt` but KV actually at `prompt + 1`) breaks RoPE: new request's tokens get attention positions off-by-one for every subsequent decode step.

**Rule**: Two viable paths remain, BOTH larger than a single-portion budget —
- **Path A — manual prefill**: call `model(input_ids, pixel_values=..., cache=cache)` directly to populate the cache yourself, snapshot at exactly prompt depth, then bootstrap `stream_generate` with a 1-token kickoff.  Requires reproducing pixel_values / mask / position_ids handling correctly against mlx-vlm's internal conventions.
- **Path B — wrap `stream_generate_vlm`**: intercept between prefill completion and first decode by patching the generator internals.  Fragile to upstream changes.

When this is next attempted: read mlx-vlm source first, open a dedicated design doc, treat G4 as a feature, not a themed cleanup.  Do not merge a half-fix that inserts a technically-unusable prompt-only entry — it'll waste cache slots and muddy the P4 metrics.

### Canonical/Model Math Consistency + Template-Kwarg Robustness (Portion 6, 2026-04-24)
Three small tightenings, all localized, shared regression surface:
- **`matched_prefix_len` is canonical-space**: the full-cover path used to subtract 1 (a model-space bootstrap-trim accounting leak) in the trimmable branch but not the non-trimmable branch, and the exact-key fast path also subtracted 1.  All three now return the canonical match length without the -1 adjustment.  The trim-by-1 is an implementation detail of the decode bootstrap, not a user-visible match length.  Telemetry and M3 threshold comparison are now consistent across paths.
- **`_cull_redundant_prefixes` must also check model_tokens**: pre-P6 the cull used canonical-prefix relation alone.  G1's markers already prevent cross-image canonical aliasing, but the invariant is "never delete an entry whose model stream is distinct from the new one."  Cull now accepts `new_model_tokens` and requires `entry.model_tokens` to be a prefix of it before culling.  Defense-in-depth against any future normalisation pass that re-introduces cross-stream canonical aliasing.
- **`preserve_thinking` kwarg is probe-gated**: pre-P6 forwarded unconditionally; a stricter-kwarg transformers/mlx-vlm upgrade would TypeError every request.  A single startup probe (`_probe_preserve_thinking_support`) on the tokenizer + processor + processor.tokenizer targets sets `_PRESERVE_THINKING_SUPPORTED`; `_chat_template_extras` only forwards when True.  Zero per-request cost; defensive posture.

**Rule**: Canonical-space and model-space token counts are not interchangeable.  Any reported "match length" must name its space.  Any cull/dedup across cache entries must check both spaces.  Any kwarg forwarded to external template code must be capability-probed at startup, not assumed.

### DFlash Speculative Decoding Was Removed (2026-04-24)
mlx-vlm's DFlash drafter was integrated speculatively in an earlier phase.  Measured on Qwen3.6-35B-A3B-UD-MLX-4bit (hybrid VLM, our primary model) it delivered a NET PERFORMANCE REGRESSION — the draft/target acceptance dynamics on the hybrid ArraysCache+KVCache architecture are incompatible with the accept-rate assumptions in DFlash's design.  Rather than carry harmful dead code + a threading-compatibility note for a feature with negative value, the full integration was removed: import, Settings fields, startup loader, `_stream_generate_unified` branch, module global.  Git history preserves the implementation; `.env.example` carried no residual vars (already clean).

**Rule**: Speculative optimizations ship with an "off" default AND a kill-switch for exactly this reason — so when empirical measurements turn negative we can remove them cleanly without multi-week regression cycles.  If DFlash is ever reconsidered, verify accept-rate on the specific target model FIRST before adding integration code.

### bf16→fp16 Conversion Breaks MLX Quantized Models
Confirmed 2026-03-21: converting bf16 scales/biases to fp16 in mlx-community Q4 quants (Qwen3.5-27B-4bit, Qwen3.5-35B-A3B-4bit) produces garbage output. Model loads without error, tensors are numerically equivalent, but the model generates random tokens. MLX's quantized dequantization kernels depend on the scales dtype being bf16 — changing it to fp16 disrupts the Metal kernel compute path. The "bf16 emulation overhead on M1" claim from external optimization reports does not apply to MLX quantized inference.

**Rule**: Never convert bf16→fp16 in quantized model weights. The dtype is load-bearing for MLX's quantized matmul kernels, not just a precision choice.

## MLX Cache API (confirmed)

| Operation | Works? | Notes |
|---|---|---|
| `trim_prompt_cache(cache, n)` | Yes | Tail-only, subtracts from `.offset` |
| `KVCache.state=` setter | Yes | Accepts arbitrary tensors, sets offset |
| Segment surgery via concatenate | Yes | Works on mock data, but RoPE invalidates suffix |
| Non-contiguous block reuse | No | RoPE bakes absolute position into K vectors |

## Client Behavior (confirmed from real traffic)

| Client | Tool results in history? | Mid-history injection? | System message stable? | Volatile fields? |
|---|---|---|---|---|
| OpenCode | No -- stripped before forwarding | No -- pure tail-append | Yes | None at per-message level |
| OpenClaw | Yes -- as `toolResult` messages | Yes (FP-2 scenario) | No -- Inbound Context block varies | `[Day YYYY-MM-DD HH:MM TZ]` frozen at send time (not volatile) |
| OpenClaw sub-agent | Stats runtime field is volatile | Unknown | Different system message per agent | `Stats: runtime Xm Ys` (normalised by Fix D) |
