# Phase 4-5: OOM Prevention & Context Lobotomy Fix (COMPLETE)

Completed: 2026-03-14

## Phase 4 -- OOM Prevention & Cache Correctness

Motivated by real-session OOM crash (Metal kIOGPUCommandBufferCallbackErrorOutOfMemory) during 20k+ token OpenClaw session.

### Fixes
1. [x] `_delete` now calls `mx.metal.clear_cache()` -- evicted KV tensors no longer linger in Metal pool
2. [x] `PROMPT_CACHE_MAX_ENTRIES_GLOBAL` reduced from 64 to 16 (was 167 GB theoretical)
3. [x] `_insert_cache_entries` deepcopy gated to `tool_calls or is_vlm` only (was every turn)
4. [x] `fetch_nearest_cache` longest-candidate selection (was arbitrary set iteration -- caused 2.6% hit instead of 93%)

### Known limitations
- L7: Heartbeat/side-session collision mitigated but not eliminated (longest-candidate wins)
- L8: `mx.metal.clear_cache()` on every `_delete` adds small sync cost (negligible)

## Phase 5 -- Critical Bug: Context Lobotomy via Normalisation Regex

### Root cause
`INBOUND_TRUSTED_CONTEXT_BLOCK_PATTERN` used `re.DOTALL` with `.*?` that scanned across entire `# Project Context` section. When pre-normalised content had no immediate ` ```json `, the regex consumed 17,835 chars (AGENTS.md, SOUL.md, TOOLS.md, USER.md, HEARTBEAT.md, MEMORY.md). Model received 25,731 chars instead of correct 43,473.

### Fix
Removed `.*?` lookahead. Pattern requires ` ```json ` immediately after header.

### Follow-up bug
Fix was too strict -- OpenClaw has 3-4 description lines between header and ` ```json `. Pattern never matched, causing duplicate block injection. Relaxed to allow up to 10 lines, anchored by negative lookahead to prevent crossing section boundaries.

### Lesson
This is the origin of the "NO re.DOTALL" rule. Led directly to Phase 6 (dual pipeline architecture).
