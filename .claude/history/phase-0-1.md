# Phase 0-1: Instrumentation, Reproduction & Architecture Design (COMPLETE)

Completed: 2026-03-13

## Phase 0 -- Instrumentation & Reproduction

### 0.1 -- Existing logging infrastructure
- [x] Audited log format: full messages array, token counts, cache hit/miss decision, block matches
- [x] Identified log entries for known bad sessions (ground-truth failure traces)
- [x] Built `scripts/diff_turns.py`: message-level structured diff (JSON + pretty-print)

### 0.2 -- Controlled reproduction
- [x] Defined minimal reproduction scenario (short session triggering large cache miss)
- [x] Built `scripts/probe_session.py`: scripted HTTP traffic generator, 4 scenarios
- [x] Defined anomaly thresholds: >30% hit drop or <20% absolute on turn >3
- [x] Captured 3+ distinct failure patterns

### 0.3 -- Failure pattern taxonomy
- [x] FP-1: Early Drift (trailing whitespace in system message -> 0% hit)
- [x] FP-2: Mid-History Insertion (tool result spliced mid-history -> partial hit)

## Phase 1 -- Architecture Design

### Candidates evaluated
- **A -- Message-level structural diff + stable prefix**: ADOPTED. Diffs message list, finds longest identical prefix, trim KV to that boundary, re-prefill rest.
- **B -- Partial KV cache substitution (surgery)**: DEFERRED. Physically possible but RoPE invalidates suffix reuse.
- **C -- Normalisation-first pipeline**: ADOPTED as complement to A.
- **D -- Speculative prefill**: REJECTED. High complexity, A+C sufficient.

### Key findings
- `trim_prompt_cache` is tail-only. `KVCache.state=` enables surgery but RoPE makes suffix reuse invalid.
- Cold prefill: ~240 TPS on Qwen3.5-35B (475 tokens). ~42s per 10k token re-prefill.
- Stable caching is critical for usable latency.

### Decision: Adopt A+C, Defer B, Reject D.
Architecture: Message-Aware Stable-Prefix Cache with dual-pipeline normalisation.
