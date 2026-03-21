# Lessons Learned

Hard-won knowledge from Phases 0-8. Read this before touching cache logic.

## Failure Patterns (confirmed, documented)

| ID | Name | Trigger | Impact | Mitigation |
|---|---|---|---|---|
| FP-1 | Early Drift | Trailing whitespace, timestamp, or `<system-reminder>` change in system message | 0% hit (total miss) -- hash chain breaks at block 0 | `_normalize_message_content_for_diff()` strips whitespace; `_canonicalize_messages()` handles volatile fields |
| FP-2 | Mid-History Insertion | Tool result spliced mid-history (OpenClaw `toolResult` messages) | Partial hit (prefix only) -- hash chain breaks at insertion point | `SESSION_TURN_STORE` + `_stable_prefix_token_len()` recovers prefix KV |

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
