# Phase 8: Multi-Agent Sub-Session Cache Collapse (COMPLETE)

Completed: 2026-03-14

## Context

Real OpenClaw multi-agent session (CEO -> Spock -> Dave sub-agent). Spock's turn responding after Dave's completion hit 50.4% cache instead of expected ~98%.

### Session structure (from logs)
| Turn | Agent | Tokens | Cache | Notes |
|------|-------|--------|-------|-------|
| T1 | Spock | 15048 | 0% miss | System msg 44,059 chars |
| T2 | Spock | 15421 | 97.6% | Same system msg |
| T3 | Dave | 11347 | 4% | Different agent, 23,031 char system msg |
| T4-T6 | Dave | ~11.5k | 98-99% | |
| T7 | Spock | 15717 | **50.4%** | System msg 43,478 chars (Inbound Context absent) |

All 7 requests used same implicit session ID (shared first ~455 tokens).

## Root Causes & Fixes

### RC1 -- System message differs (T2: 44,059 chars vs T7: 43,478 chars)
Inbound Context block (581 chars) present in T1/T2, absent in T7. Shift breaks 7,534 tokens.
**Fix B**: `_canonicalize_inbound_context_block()` produces identical output whether block is present or absent. Both cases -> `__STABLE_INBOUND_CONTEXT_SECTION__\n# Project Context...`.

### RC2 -- VLM path lacks canonicalization
No `_canonicalize_messages`, no `_scrub_cache_key`, no dual pipeline for VLM.
**Fix C**: VLM canonical pipeline added. `prompt_tokens` via processor CPU tokenizer (cache key), `model_tokens` via VLM processor (model input).

### RC3 -- `SESSION_MAX_IDLE_SECONDS=0` destroys SESSION_TURN_STORE
`0` meant "always expired". Every call pruned all records. Dave's record overwrote Spock's.
**Fix A**: `0` now means "never expire" (same convention as `LRUPromptCache._is_expired`).

### RC4 -- Sub-agent stats volatile field
`Stats: runtime 1m52s` changes per execution.
**Fix D**: `SUBAGENT_STATS_PATTERN` normalises to `Stats: runtime __STABLE_RUNTIME__`.

## Open follow-ups
- [ ] Regression test Phase 8 fixes (probe_session.py)
- [ ] VLM canonical pipeline validation (2-turn session)
- [ ] SESSION_TURN_STORE for VLM via canonical messages
