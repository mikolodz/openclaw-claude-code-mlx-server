# Goals & Strategic Constraints

## North Star

Build the best possible KV-cache-aware inference server for Apple Silicon.

**"Best" means:**
- Maximum cache hit rate across the full range of real client behaviour -- pure tail-appends, mid-history tool-result injections, multi-agent sub-session spawns, context compaction, heartbeats, silent system-prompt drift.
- Minimum prefill latency. On Apple Silicon with a 27B model, cold 10k-token prefill costs ~42 seconds. Cache hit rate IS the product.
- SOTA engineering quality. Design informed by vLLM, SGLang, and production inference systems. Established patterns applied to the MLX/Apple Silicon constraint set.

## Guiding Principles

1. **Evidence before code.** No caching changes without log-confirmed reproduction.
2. **Systemic over surgical.** Architecture that handles the class of problems, not each instance.
3. **Think in structures, not strings.** Diff and cache at the message-struct level, not serialised strings.
4. **Reuse as much KV state as physically possible.** Even if middle changed, the prefix to that point is golden.
5. **Observability is first-class.** If we can't measure it precisely, we can't improve it.

## Constraints

| Rule | Detail |
|---|---|
| No model blinding | All semantic content always reaches the model intact |
| No gut-feel patches | Every code change must cite a specific log/diff |
| No cross-session leakage | Cache isolation between sessions maintained |
| Single-file architecture | Changes stay in `start-llm.py` unless explicitly agreed |
| Python 3.12 only | No 3.13+ syntax |
| Thread safety | All shared state protected by locks |
| Normalisation is pre-hash only | Model always sees real content |
| Agility | Must handle arbitrary clients without per-client patches |

## Active Goals (prioritised, 2026-04-24)

Ordered by priority.  G1–G3 are true bugs uncovered while fixing the VLM
cache regression (see `.claude/history/phase-9-vlm-fix.md` once written) and
are the direct next step.  G4 is the quality follow-up that closes the last
cache-loss path for hybrid VLMs.  G5–G7 are themed groupings so no open
issue is lost.  G8–G9 are the SOTA performance/capacity upgrades we want to
land once correctness is settled.

### G1 (P0, BUG, NEXT STEP) — Image-identity cache correctness
**Problem.** `image_token_id=248056` on Qwen3.6 is a single id reused 3120× per
image.  Model-space LCP compares token ids, so image A's KV gets reused for a
request carrying image B at the same canonical position — no crash, silently
wrong answers for any screenshot-swapping workflow (pi, OpenClaw multi-turn
with new images).
**Approach.** Mix a SHA-256 of each `image_url` payload (in message order)
into the canonical cache key — either by inserting a stable synthetic token
per image at canonicalisation time, or by hashing payloads into the cache
lookup key post-tokenisation.  Image content change becomes visible at the
token stream and causes a clean miss.

### G2 (P0, BUG, NEXT STEP) — Generation watchdog + `model_lock` recovery
**Problem.** No per-request deadline.  A Metal stall, infinite `<think>`
loop, or pathological sampler holds `model_lock` forever; every subsequent
request queues until SIGTERM.  Currently only recoverable by killing the
server.
**Approach.** Wall-clock deadline that aborts the generator iterator, evicts
any partial cache entry that would otherwise be inserted with incomplete
KV state, and unconditionally releases `model_lock` in the finally path.

### G3 (P0, BUG, NEXT STEP) — Canonical-form write guard for `SESSION_TURN_STORE`
**Problem.** `_update_session_turn_store`'s write guard compares incoming
messages to the stored record via `_normalize_message_content_for_diff`,
which only strips whitespace.  OpenClaw drifts per-turn on `message_id`,
`Inbound Context` block, sub-agent `Stats: runtime`, timestamps — any single
drifted field and the guard refuses to update, so M3 stable-prefix recovery
stops advancing for the remainder of the session.
**Approach.** Run the guard comparison on `_canonicalize_messages` output
(the same canonical form the cache key is derived from) so cache-irrelevant
drift is treated as equal; the turn record keeps advancing across tool-
result insertions and OpenClaw metadata shuffles.

### G4 (P1, quality) — Pre-generation KV checkpoint for hybrid VLMs
**Problem.** Prompt-only cache checkpoint insertion is gated on
`can_trim_prompt_cache`, which returns False for Qwen3.6 hybrid
(ArraysCache layers aren't trimmable).  The branch never fires for VLM.  If
the healing-hash path ever misses (different client serialisation, content
normalised client-side) cache reuse collapses to zero for the whole
session.
**Approach.** Deep-copy `prompt_cache` **before** `stream_generate` extends
it with generated tokens; insert that as the prompt-only entry directly,
removing the `can_trim` dependency.  Likely requires a small refactor of
the generation call-site so we can hook between "prefill complete" and
"generation starts".

### G5 (P2, themed) — Cache lookup performance & eviction policy
**Covers.** LCP O(n·k) cost per lookup; `_cull_redundant_prefixes` deleting
useful prompt-only checkpoints; fixed `PROMPT_CACHE_MAX_ENTRIES_GLOBAL=16`
regardless of actual GPU memory pressure; the partial-image retreat in
`do_POST` being redundant overhead once G1 lands.
**Approach.** Chain-hash `model_tokens` so LCP only runs inside
block-matched candidates (saves 0.3M–1.6M int comparisons per lookup at
20k–100k context).  Teach eviction to prefer keeping prompt-only entries
over long but stale full-turn entries.  Size the cache dynamically from
`mx.metal.get_active_memory()` against a headroom target.  Remove the
partial-image retreat once G1's tests are green.

### G6 (P2, themed) — Cache-key derivation robustness
**Covers.** `_scrub_cache_key` runs regex on the rendered canonical string;
a future pattern could accidentally match inside a
`<|vision_start|>…<|vision_end|>` span and desync the cache key from the
model stream.  `_compute_msg_token_boundaries` falls back to
equal-distribution for VLM, leaving M3 stable-prefix token math noisy.
`trim_prompt_cache` crossing `QUANTIZED_KV_START=16000` is untested.
`vlm_kwargs` (image_grid_thw, etc.) is still forwarded to mlx-vlm even when
`pixel_values=None` — undefined behaviour territory.
**Approach.** Mark and skip vision spans in `_scrub_cache_key`; reuse the
CPU text tokenizer for VLM per-message token boundaries (same trick the
canonical-key pipeline already uses); add regression tests for trim across
the quant boundary; drop image-only kwargs on pure-text prefill calls.

### G7 (P3, themed) — Multi-process / ops hygiene
**Covers.** `CACHE_SESSION_PARTITIONING=true` is read but never applied in
`select_best_cache` (multi-tenant prefix leakage risk).  `HEALING_STORE` is
2000-entry in-memory only — lost on restart, not shared across processes.
`mx.metal.clear_cache()` is deprecated; MLX logs warn, and the next upgrade
will remove it.  LiteLLM proxy adds a TCP hop per request now that our
`Connection: close` fix disables keep-alive pooling.
**Approach.** Wire session partitioning through the primary lookup for
multi-tenant isolation; move healing to SQLite or a file-backed LRU;
migrate to `mx.clear_cache()`; evaluate removing LiteLLM by handling its
`drop_params` + `reasoning_effort` mapping directly in `do_POST`.

### G8 (P2, capability) — Port oMLX SpecPrefill
**Problem.** Cold prefill on Qwen3.6 35B costs ~11s per 3.7k tokens (~325
tok/s).  oMLX's SpecPrefill implementation gets ~2× prefill throughput by
running a draft model in parallel on the prefill path.
**Approach.** Port `_qwen36_extract_queries` and the SpecPrefill orchestration
from `jundot/omlx` (see `.../omlx/patches/specprefill.py`).  **Only after
G1–G3 land** — running SpecPrefill on top of a cache that silently serves
stale image KV is negative value.  Must preserve the dual-pipeline invariant
(model always sees original tokens).

### G9 (P2, capability) — Paged SSD KV cache tier
**Problem.** RAM cache saturates 96–99% hit rate but `max_entries=16` at
~2–3 GB/entry is GPU-memory bound.  Long multi-hour sessions or many
parallel tenants overflow and thrash.  oMLX demonstrates 85–90% cache-read
throughput using paged SSD backing (see `.../omlx/cache/paged_ssd_cache.py`)
but we noticed looping / context poisoning / agent lobotomisation during
their real-traffic runs — so a straight copy is not acceptable.
**Approach.** Design a dedicated SSD tier behind the RAM LRU using
block-level page-out/page-in, keep model-space LCP correctness from G1–G4,
and NOT import oMLX's full cache algorithm (it has suspected correctness
bugs).  Must land after G1–G4 are green so we extend a correct cache, not
a broken one.  Produce a design doc in `.context/` before implementation.

## Closed Goals

- **Fix VLM cache regression** (closed 2026-04-24).  Image + tool-call path
  hit 94–99% on pi sessions; zero `Image features and image tokens do not
  match` crashes; cross-session contamination eliminated.  See TASKS.md
  "Recently Closed" for the full fault list and fix summary.
