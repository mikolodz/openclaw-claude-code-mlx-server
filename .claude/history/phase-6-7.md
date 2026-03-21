# Phase 6-7: Dual Pipeline & Post-Audit Fixes (COMPLETE)

Completed: 2026-03-14

## Phase 6 -- Dual-Pipeline Normalisation Architecture

Root cause of every normalisation bug: cache key and model input were the same string, mutated in-place. The fix is architectural: split into two separate pipelines at message-struct level.

### Steps completed
1. [x] `_canonicalize_messages()` introduced -- returns `(original, canonical)` deep copies
2. [x] LM path wired: original -> model_tokens, canonical -> prompt_tokens (cache key)
3. [x] VLM path: removed `_normalize_prompt_for_cache` calls
4. [x] Renamed `_normalize_prompt_for_cache()` -> `_scrub_cache_key()`, removed DOTALL patterns
5. [x] Safety invariant `_assert_cache_key_safety()` added (>= 90% length check)
6. [x] Double normalisation eliminated from VLM path
7. [x] Telemetry: `cache_key_normalized`, `cache_key_delta_chars`

### Remaining
- [ ] Regression test against Phase 5 log (Step 5)
- [ ] Probe session validation (Step 5)

## Phase 7 -- Post-Audit Fixes

### Fix 1 -- Token boundary mismatch (CRITICAL)
`matched_prefix_len` is canonical token count. Using it to slice `model_tokens` (original) produces wrong RoPE positions when canonical != original length. Fixed: `_kv_cache_offset(cache)` reads actual KV depth from `cache[0].offset`.

### Fix 2 -- Orphaned DOTALL regex constants
Deleted `INBOUND_TRUSTED_CONTEXT_BLOCK_PATTERN`, `CACHE_STABLE_INBOUND_CONTEXT_BLOCK`, `INBOUND_CONTEXT_TO_PROJECT_BOUNDARY_PATTERN`. Left tombstone comment.

### Fix 3 -- `_canonicalize_inbound_context_block` closing fence
Fixed `content[fence_close_end:]` -> `content[fence_close_start:]` to preserve closing ` ``` ` fence.

### Fix 4 -- Misleading alias
Deleted `_normalize_prompt_for_cache = _scrub_cache_key` alias. No remaining call sites.

### Validation
All probe_session.py scenarios at Phase 3 baseline. No `rest_tokens` site uses canonical count as raw index.
