# PLAN.md — SOTA Cache Architecture for Local MLX LLM Server

This file is both a living design document and a checklist.
Check off items as they are completed. Do NOT patch code on gut feeling —
every change must be grounded in observed, logged evidence.

---

## Problem Statement

The server runs locally on Apple Silicon. RAM is plentiful; **prefill is the bottleneck**.
The existing cache architecture (block-based hash chains, similar to Radix tree / vLLM)
is solid in design but breaks in practice because clients mutate the prompt mid-history.

Two primary offenders:

1. **Tool-result injection with a delay** — OpenCode / OpenClaw send turn N+1 with the
   tool result from turn N spliced somewhere in the middle of the prior conversation.
   The context is not a clean append; it's a mid-history insertion.
2. **Silent history mutations** — a new user message can arrive with slightly different
   prior turns: timestamps changed, keepalive pings injected, thinking blocks
   reordered, system-prompt drift, metadata fields added/removed.

The result: cache misses that look catastrophic — 0% hit in a long live session, or a
sudden drop to ~40% hit after a single tool call in an otherwise stable conversation.

**This is not a simple bug. Weeks of expert effort and top-tier AI agents (Gemini,
GPT, etc.) with a patch-by-patch approach failed to stabilise it. The approach changes
here: diagnose first, understand the full landscape, then design a systemic solution.**

**Hard constraint — no model blinding:** The LLM must receive the full, real content
at all times — tool results, `<think>` blocks, everything. We cannot hash, skip, stub,
or otherwise obscure semantic content just to get a cache hit. A cache win that
blinds or lobotomises the model is worthless by definition.

---

## Context & History

Read this before touching anything. It will save you time.

This codebase is the result of weeks of serious, focused engineering. The prompt
caching system is not naive — it uses a block-based hash chain architecture
conceptually similar to Radix trees and vLLM's approach. It was designed carefully.
The problem is not that the architecture is wrong in theory. The problem is that
the clients (OpenCode, OpenClaw) behave in ways that are structurally hostile to
any prefix-based cache: they mutate history mid-conversation, inject tool results
with delays, and silently swap earlier turns when a new user message arrives.

**What has already been tried and failed:**
Multiple weeks of iteration by experienced developers, assisted by top-tier AI
agents (Gemini, GPT-class models). The approach was: observe a cache miss, identify
a specific cause, add a normalisation rule to patch it, repeat. This did not work.
Each patch fixed one symptom and exposed or introduced another. The system became
harder to reason about with every patch, and the fundamental miss rate did not
improve in a durable way.

**Why the approach is changing:**
The patch-by-patch method fails because it treats each cache miss as an isolated
bug rather than as a symptom of a structural mismatch between the cache architecture
and the client's behaviour. We need to understand the full landscape of failure modes
first, then design a system that handles the *class* of problems — not a growing
list of special cases.

**The emotional weight behind "no gut-feel patches":**
If a change to caching logic feels obvious or intuitive, that is a warning sign, not
a green light. The previous weeks produced many "obvious" fixes that didn't hold.
The only thing that can be trusted here is a log entry that shows a specific,
reproducible failure, a diff that isolates exactly what changed in the prompt, and
a design that addresses the structural cause. Everything else is noise.

**The north star:**
A session where OpenCode or OpenClaw runs 10+ turns with tool calls, and the cache
hit rate stays high and stable throughout — not because we hid anything from the
model, but because the architecture correctly identifies what is genuinely stable
in the context and reuses it.

---

## Guiding Principles

1. **Evidence before code.** No change to caching logic without a log-confirmed
   reproduction of the specific failure mode it targets.
2. **Systemic over surgical.** We are not looking to patch one normalisation rule at
   a time. We want an architecture that handles the class of problems, not each
   individual instance.
3. **Think in structures, not strings.** The prompt is a sequence of typed messages
   with roles, content blocks, and metadata. Diff and cache at that structural level,
   not on serialised strings.
4. **Reuse as much KV state as physically possible.** Even if the middle of history
   changed, the prefix up to that change point is golden — protect it.
5. **Observability is a first-class feature**, not an afterthought. If we can't
   measure it precisely, we can't improve it.

---

## Phase 0 — Instrumentation & Reproduction (DO THIS FIRST)

> Goal: build the tooling to catch, record, and replay cache-miss events. No
> architecture changes until Phase 0 is complete and at least two distinct failure
> modes are confirmed with real traces.

### 0.1 — Leverage existing logging infrastructure

The server already writes per-request logs to `logs/<YYYY-MM-DD>/cache-session-<hash>.log`.
These logs contain the full incoming prompt and cache metrics. This is our primary
evidence base — we must use it before writing a single line of fix code.

- [x] **Audit the existing log format** — confirm exactly which fields are captured:
      full messages array, token counts, cache hit/miss decision, which blocks matched,
      which blocks were dropped.
- [x] **Identify log entries for known bad sessions** — find real log files where a
      dramatic cache drop occurred (0% or large regression mid-session). These become
      our ground-truth failure traces.
- [x] **Write a log-diff tool** (`scripts/diff_turns.py`) — given two consecutive log
      entries from the same session, produce a structured diff at the message level:
      which messages were added, removed, or mutated, and where in the sequence.
      Use Python's `difflib.SequenceMatcher` or a custom message-aware differ.
      Output must be human-readable and machine-parseable (JSON + pretty-print).

### 0.2 — Controlled reproduction via sub-agents

To catch failures on demand rather than waiting for them in the wild:

- [x] **Define a minimal reproduction scenario** — a short OpenCode or OpenClaw
      session that reliably triggers a large cache miss. E.g.: start session, send a
      message that causes a tool call, observe cache hit rate on the follow-up turn.
- [x] **Build a sub-agent harness** (`scripts/probe_session.py`) — spawns OpenCode
      (or simulates its HTTP traffic) against the local server with a scripted
      conversation, captures all request/response pairs, and records the incoming
      `messages` arrays turn by turn.
- [x] **Define "anomaly" thresholds** — a cache event is worth logging for analysis
      when: (a) hit rate drops >30% compared to previous turn, or (b) absolute hit is
      <20% on a session turn >3. When an anomaly fires, the harness dumps a full
      structured diff automatically.
- [x] **Run the harness until at least 3 distinct failure patterns are captured** —
      do not proceed to Phase 1 until we have concrete, repeatable examples.

### 0.3 — Failure pattern taxonomy

Once raw diffs are in hand, classify each observed failure into a named pattern:

- [x] **Compile a failure taxonomy** in this file (add a section below) — each
      entry: pattern name, example diff snippet, frequency estimate, which client
      triggers it, and what the cache impact is.

---

## Phase 1 — Landscape Analysis & Architecture Design

> Goal: understand the full solution space before committing to any implementation.
> No code written in this phase — only design documents and decisions.

### 1.1 — Map the current cache invalidation landscape

- [x] **Trace every cache-flush trigger in `start-llm.py`** — for each code path that
      discards or rebuilds `PROMPT_CACHE`, document: the condition, the line number,
      the intended reason, and whether it fires on any of our captured failure traces.
- [x] **Determine what MLX's cache API can and cannot do** — confirmed by reading mlx_lm
      source and running targeted mock tests (scripts/inspect_mlx_cache.py):
      - `trim_prompt_cache` / `KVCache.trim()` is **tail-only** (subtracts from `.offset`).
      - `KVCache.state` setter accepts any `[B, heads, T, head_dim]` tensor pair →
        segment surgery via `mx.concatenate + state=` is **confirmed physically possible**.
      - `QuantizedKVCache` (used with KV_BITS set): surgery requires dequantize → splice →
        re-quantize → `state=` + manual `.offset` update. Feasible but lossy.
      - **RoPE caveat**: position encodings are baked into K vectors — reuse is only
        valid for blocks at the same absolute token positions.
- [x] **Measure the cost of re-prefill at various depths** — real log evidence from
      probe_session runs (logs/2026-03-13/):
      - Cold prefill (475 tokens): 240 TPS (`prefill_tps: 240.8`, session 57f9f084)
      - Warm prefill with cache hit (32 rest tokens): 41 TPS (same session, turn 2)
      - Small cold prompt (29 tokens): 15–63 TPS depending on prior GPU state
      - At 240 TPS, a 10k token re-prefill costs ~42 seconds wall-clock.
      - **Confirmed: stable caching is critical for usable latency.**

### 1.2 — Design candidates

Evaluate each candidate design against the failure taxonomy from Phase 0:

- [x] **Candidate A — Message-level structural diff + longest stable prefix**
      ADOPTED. Diffs message list at boundaries; finds longest normalisation-identical
      prefix; reuses KV up to that prefix via trim; re-prefills the rest. Handles both
      FP-1 and FP-2. Does not require segment surgery (tail-trim only). See DDR.md.

- [x] **Candidate B — Partial KV cache substitution (segment surgery)**
      DEFERRED. Segment surgery is physically possible (KVCache.state= setter confirmed
      via mock tests). However, RoPE position encoding invalidates suffix KV reuse after
      any mid-history insertion: the suffix messages shift absolute positions, making
      their cached K vectors incorrect. Surgery is only safe for the prefix — which
      Candidate A already handles via trim. No benefit over A without RoPE recomputation.

- [x] **Candidate C — Normalisation-first pipeline (complementary to A)**
      ADOPTED. Per-message normalisation applied before diff computation. Strips trailing
      whitespace, `<system-reminder>` blocks, timestamps. Widens the stable prefix window.
      Original content always reaches the model intact. See DDR.md.

- [x] **Candidate D — Speculative prefill**
      REJECTED for Phase 2. High complexity; A+C resolve the primary failure patterns.
      Revisit after Phase 3 validation only if needed.

- [x] **Write a design decision record (DDR)** — DDR.md written with full evidence base,
      candidate evaluation, confirmed failure patterns, MLX API findings, and Phase 2
      implementation contract (M1–M6).

---

## Phase 2 — Implementation

> Only begins after Phase 1 DDR is signed off.
> All implementation steps reference specific failure patterns from the taxonomy.

- [x] **M1 — Diff infrastructure**: `_message_diff(prev, curr)` implemented in
      `start-llm.py`. Returns `stable_prefix_count` (leading equal messages) and
      change descriptors. Unit-tested against FP-1 and FP-2 patterns — all passing.
- [x] **M2 — Stable-prefix cache reuse**: `_stable_prefix_token_len()` + `SESSION_TURN_STORE`
      implemented. Cache lookup in `do_POST` now does a secondary stable-prefix lookup
      if the global cache match is weaker than the message diff says it could be.
- [x] **M3 — Mid-history insertion handling**: handled via M2 — when an insertion is
      detected at position K, only messages [0..K-1] are counted as stable; the lookup
      uses only those tokens as the prefix floor.
- [x] **M4 — Normalisation pipeline v1**: `_normalize_message_content_for_diff()`
      implemented; currently strips trailing/leading whitespace (confirmed FP-1 trigger).
      Full M4 (extended patterns for real OpenCode/OpenClaw drift) is marked incomplete
      pending real session log audit — see DDR.md caveat.
- [x] **M4 extended — Real-traffic normalisation audit**: see Phase 3 entry below for full
      findings. Conclusion: no per-message-level volatile fields observed in OpenCode sessions.
      No additional normalisation rules needed for current OpenCode + Qwen3.5 setup.
      OpenClaw, context compaction, and long sessions still untested.
- [x] **M5 — `SessionTurnRecord` storage**: `_update_session_turn_store()` called inside
      both `_insert_cache_entries` blocks (streaming and non-streaming paths). Stores
      messages + per-message token lengths after each successful turn.
- [x] **M6 — Telemetry enhancements**: prompt log now includes `stable_prefix_msg_count`,
      `stable_prefix_token_len`, and `stable_prefix_diff` (capped at 8 descriptors) for
      every request. Observable in `logs/<date>/cache-session-*.log`.
- [ ] **M5 (Candidate B) — Cache segment surgery**: DEFERRED per DDR.md (RoPE position
      correctness invalidates suffix KV after mid-history insertion).

---

## Failure Pattern Taxonomy

*To be populated during Phase 0.3. Each entry added here as patterns are confirmed.*

| ID | Name | Client | Description | Cache impact |
|---|---|---|---|---|
| FP-1 | Early Drift (Silent Mutation) | OpenCode/Claw (Simulated) | A small change (e.g., whitespace, timestamp) in an early message (like System Prompt) invalidates the hash chain for all subsequent blocks because hash depends on prefix. | 0% Hit Rate (Total Miss) |
| FP-2 | Mid-History Insertion | OpenCode/Claw (Simulated) | Inserting a message (e.g., missed tool output) in the middle of history breaks the hash chain at the insertion point. Prefix is preserved, suffix is lost. | Partial Hit (Prefix Only) |

---

## Phase 4 — OOM Prevention & Cache Correctness (2026-03-14)

> Motivated by real-session OOM crash (Metal kIOGPUCommandBufferCallbackErrorOutOfMemory)
> during a 20k+ token OpenClaw session with repeated cache divergence.

### Root causes identified

| # | Problem | Location | Status |
|---|---------|----------|--------|
| 1 | `_delete` never called `mx.metal.clear_cache()` — evicted KV tensors lingered in Metal pool, causing monotonic GPU memory growth | `LRUPromptCache._delete` line ~660 | **Fixed** |
| 2 | `PROMPT_CACHE_MAX_ENTRIES_GLOBAL=64` in `.env` — at 20k tokens × 2.6 GB/entry = 167 GB theoretical max, far exceeding 64 GB RAM | `.env` | **Fixed → 16** |
| 3 | `_insert_cache_entries` unconditionally `copy.deepcopy`-ed the full KV tensor on **every** turn to create a prompt-only checkpoint — doubled GPU pressure on non-tool turns | `_insert_cache_entries` line ~1921 | **Fixed — gated to `tool_calls or is_vlm` only** |
| 4 | `fetch_nearest_cache` iterated a Python `set` of candidates and took the **first match regardless of length** — when a short heartbeat/branch entry and the long conversation entry shared the same block hash bucket, the short one could win, causing a near-total cache flush (2.6% hit instead of 93%) | `LRUPromptCache.fetch_nearest_cache` block-walk loop | **Fixed — now picks longest candidate at each block level** |

### Fix 4 detail — longest-candidate selection in `fetch_nearest_cache`

The block hash index maps `(model, chain_hash) → set[token_tuple]`. Multiple entries can share
the same chain hash for early blocks if their token prefixes are identical (e.g. a 10k-token
heartbeat session and a 34k-token conversation with the same system prompt). The old code broke
out of the inner loop on the first matching candidate — set iteration order is arbitrary in
CPython, so the heartbeat entry could be returned instead of the conversation entry.

Fix: scan all candidates at each block level and keep the one with the most tokens. Longest
entry = most cached context = always the right choice when multiple entries are valid prefixes.

### Debug noise reduction

The `_debug_token_divergence` block (controlled by `VLM_CACHE_DEBUG=true`) previously fired on
every `shorter` cache hit, which is the normal state for every turn in a long session (stored
entry ends at previous turn boundary, new prompt is longer). This generated alarming
`🚨 CACHE DIVERGENCE` output on every request. Fixed to only fire when the divergence is
genuinely unexpected: full miss, or matched prefix more than 64 tokens shorter than the
stable-prefix estimate.

---

## Constraints & Non-Goals

| Rule | Detail |
|---|---|
| No model blinding | Tool results, `<think>` blocks, all semantic content always reaches the model intact |
| No gut-feel patches | Every code change must cite a specific log entry or diff that motivated it |
| No cross-session leakage | Cache isolation between sessions must be maintained |
| Single-file architecture | Changes stay in `start-llm.py` unless a refactor is explicitly agreed |
| Python 3.12 | No 3.13+ syntax |
| Thread safety | All new shared state protected by appropriate locks |
| Normalisation is pre-hash only | Normalised form used only for cache key/diff; model always sees real content |

---

## Open Questions

- ~~Does MLX's `trim_prompt_cache` / `can_trim_prompt_cache` support arbitrary interior
  trim, or only tail trim?~~ **Answered:** tail-only. `KVCache.state=` enables surgery
  but RoPE makes suffix reuse invalid. Candidate B deferred. (2026-03-13)
- ~~What is the real prefill throughput on this machine?~~ **Answered:** ~240 TPS cold
  on Qwen3.5-35B for 475 tokens. ~42 seconds per 10k token re-prefill. (2026-03-13)
- Can the LiteLLM proxy layer usefully pre-normalise before the MLX server sees the
  request, or does it lack the session context needed to make that decision?
  (Low priority — M4 normalisation at the MLX server level is sufficient.)
- Are there MLX cache primitives we haven't yet used (e.g. cache serialisation to
  disk, explicit block pinning) that could help? (Not needed for Phase 2.)

---

## Sub-Agent Job Descriptions

When it is time to execute sub-agent work, these are the scoped jobs:

| Job | Description |
|---|---|
| `log-auditor` | Reads existing session logs, identifies anomalous cache-miss events, extracts the turn pairs as JSON |
| `diff-tool-builder` | Implements `scripts/diff_turns.py` — message-level differ with JSON + pretty output |
| `probe-harness-builder` | Implements `scripts/probe_session.py` — scripted OpenCode/OpenClaw HTTP traffic generator |
| `taxonomy-analyst` | Given a set of diffs, classifies each into a named failure pattern and populates the taxonomy table |
| `mlx-cache-api-researcher` | Reads `mlx_lm` source to determine exactly what cache operations are possible (snapshot, trim, restore) |
| `ddr-writer` | After Phase 1, synthesises findings into the design decision record |

---

## Phase 2 — Implementation Detail (what was actually built)

> Added 2026-03-13. A fresh agent MUST read this before touching any of the new code.

### New symbols in `start-llm.py` (lines ~1047–1250)

| Symbol | Type | Purpose |
|---|---|---|
| `_SessionTurnRecord` | `@dataclass` | Stores `messages`, `msg_token_lens`, `total_prompt_tokens`, `touched_at` for one completed turn |
| `SESSION_TURN_STORE` | `Dict[str, _SessionTurnRecord]` | Global, protected by `prompt_cache_lock`. Maps `session_id → _SessionTurnRecord`. Pruned by TTL on write. |
| `_normalize_message_content_for_diff(msg)` | function | Strips leading/trailing whitespace from message content. Used ONLY for diff key computation — never modifies what the model sees. Currently minimal (whitespace only); M4 extended rules go here. |
| `_message_diff(prev, curr)` | function | Returns `(stable_prefix_count, descriptors)`. `stable_prefix_count` = number of leading messages that are normalisation-identical. Only the contiguous leading block counts — later equal blocks are ignored (RoPE correctness). |
| `_stable_prefix_token_len(session_id, curr_msgs)` | function | Reads `SESSION_TURN_STORE`, runs `_message_diff`, sums `msg_token_lens[:stable_prefix_count]`. Returns `(stable_token_len, stable_msg_count, descriptors)`. Must be called under `prompt_cache_lock`. |
| `_compute_msg_token_boundaries(messages, prompt_tokens)` | function | Computes per-message token lengths via cumulative `apply_chat_template(messages[:i+1], add_generation_prompt=False)`. Falls back to even-division for VLMs. Replaces the old `"role: content"` approximation (L1 fix). |
| `_update_session_turn_store(session_id, messages, prompt_tokens)` | function | Called after both streaming and non-streaming `_insert_cache_entries` calls, inside `prompt_cache_lock`. Writes the turn record **only if** the incoming message list is a strict append of the existing record (multi-agent write guard). |

### Integration points in `do_POST`

1. **Before cache lookup** (inside `prompt_cache_lock`): `_stable_prefix_token_len()` is called to get `stable_prefix_token_len_computed` and `stable_prefix_msg_count_computed`.

2. **After global cache lookup** (still inside `prompt_cache_lock`): if `stable_prefix_token_len_computed > matched_prefix_len`, a secondary lookup is done: `PROMPT_CACHE.fetch_nearest_cache(model_path, prompt_tokens[:stable_prefix_token_len_computed])`. If this returns a better hit, the result replaces the global lookup result. `cache_match_type` is set to `"stable_prefix_<original_type>"` and `cache_selection_source` to `"stable_prefix"`.

3. **After `_insert_cache_entries`** (inside `prompt_cache_lock`, both streaming and non-streaming): `_update_session_turn_store()` is called.

4. **Telemetry**: prompt log includes `stable_prefix_msg_count`, `stable_prefix_token_len`, `stable_prefix_diff` (first 8 descriptors).

### Known limitations and risks

**L1 — Per-message token length (FIXED 2026-03-13, revised 2026-03-13):**
`_update_session_turn_store` now uses `_compute_msg_token_boundaries()` which renders each
message prefix through `tokenizer.apply_chat_template(messages[:i+1], add_generation_prompt=False)`
to get exact cumulative token counts. **Revised fix (2026-03-13):** The last bucket is now
set to `prev_len - sum(prev[:-1])` where `prev_len` is the loop's final value — i.e. the
length of `apply_chat_template(messages, add_generation_prompt=False)`. We deliberately do
NOT use `len(prompt_tokens)` (rendered with `add_generation_prompt=True, enable_thinking=True`)
because that includes the generation-prompt suffix `<|im_start|>assistant\n<think>\n\n` (Qwen3).
Including those tokens in the last bucket caused `_stable_prefix_token_len` to sum past the
end of the real messages into the generation-prompt region. On the next request the token at
that position differed (`\n` ID 198 vs `\n\n` ID 271) because the model emits a different
`<think>` opener each time — this was the root cause of every "CACHE DIVERGENCE DETECTED AT
INDEX N" warning observed with Qwen3 + tool calls in OpenClaw. Falls back to equal-division
approximation for VLMs or if `apply_chat_template` fails.

**L2 — Secondary lookup only helps if the prior turn's cache is still alive:**
`SESSION_TURN_STORE` knows the stable prefix length, but `PROMPT_CACHE.fetch_nearest_cache` can only return something if the prior turn's KV state was stored and not yet evicted by LRU. If the cache was evicted (e.g. many concurrent sessions, or the server restarted), the stable-prefix lookup returns nothing and we fall back to a full miss. This is correct behaviour — we never synthesise a cache entry that doesn't exist.

**L3 — `SESSION_TURN_STORE` vs `SESSION_INDEX` — two parallel structures:**
`SESSION_INDEX` tracks which token-tuple keys a session has registered in `PROMPT_CACHE` (used for cache insertion bookkeeping). `SESSION_TURN_STORE` tracks the structured message list and per-message token lengths for diff purposes. They serve different roles and are intentionally separate. Do NOT merge them — `SESSION_INDEX` is keyed on token tuples; `SESSION_TURN_STORE` is keyed on session IDs with message-level structure.

**L4 — Thread safety:**
All reads and writes to `SESSION_TURN_STORE` happen inside `with prompt_cache_lock:`. Do not access it outside that lock.

**L5 — VLM path:**
The stable-prefix diff uses `messages` (the structured list post-healing). For VLM requests this is the healed+prepared message list, which is what was sent. The per-message token lengths for VLM will be less accurate (vision tokens inflate the total) but the diff correctness is unaffected — we only compare message content, not token counts, in the diff step.

**L6 — Multi-agent / sub-agent write guard (added 2026-03-13):**
`_update_session_turn_store` now checks whether the incoming message list is a strict append of the existing `SESSION_TURN_STORE` record before overwriting it. If any prior message differs, or the new list is shorter, the write is skipped. This prevents sub-agent or parallel-branch requests (which have structurally different conversation histories on the same `session_id`) from clobbering the orchestrator's record. The global `PROMPT_CACHE` block-hash lookup is completely unaffected — it is session-agnostic and serves all requests normally regardless of this guard. The stable-prefix layer is therefore safe for all multi-agent topologies; at worst it produces a conservative `stable_prefix_msg_count=0` and falls back to the global result.

---

## Phase 3 — Validation

- [x] **Start the server and run `scripts/probe_session.py` all scenarios** — confirmed
      2026-03-13. Results (logs/2026-03-13/):
      - `probe-normal`: `stable_prefix_msg_count=2`, `cache_match_type=shorter/stable_prefix_shorter` ✅
      - `probe-drift` (whitespace): `stable_prefix_msg_count=2`, `match=stable_prefix_shorter`, `sel=stable_prefix` ✅
        FP-1 (trailing whitespace) is correctly absorbed by normalisation.
      - `probe-semantic` (content change): `stable_prefix_msg_count=0`, `match=miss` ✅
        Real mutations correctly NOT absorbed (normalisation doesn't hide semantic changes).
      - `probe-insert`: `stable_prefix_msg_count=1`, `stable_tok=237` (accurate via chat-template
        boundary computation). In single-session cold run: `match=shorter (308/516)` via global. ✅
        In multi-scenario run: `match=miss` due to `_cull_redundant_prefixes` evicting T1 entry. ⚠️
        See L6 below.

      **probe_session.py was updated** (2026-03-13): the `drift` scenario was incorrectly doing a
      semantic content replace (`assistant→robot`) instead of a whitespace drift. Fixed to add a
      trailing space (FP-1 trigger). A `semantic` scenario was added to confirm normalisation
      does not absorb real changes.

      **L1 fix applied** (2026-03-13): `_update_session_turn_store` now uses cumulative
      `apply_chat_template(messages[:i+1], add_generation_prompt=False)` to compute exact
      per-message token boundaries instead of the approximate `"role: content"` tokenisation.
      `stable_prefix_token_len` for insert scenario: was 303 (over-estimated), now 237 (accurate).

      **L6 — New known limitation** (observed 2026-03-13): when running multiple probe scenarios
      sequentially, `_cull_redundant_prefixes` deletes the T1 475-token prompt-only entry when
      T2 inserts the full (505+gen)-token cache. The T2 stable-prefix secondary lookup can only
      benefit from T1's cache if T1's cache is still alive. This eviction is pre-existing
      behaviour — `_cull_redundant_prefixes` assumes longer entries supersede shorter ones. The
      probe-insert scenario hit rate is still 60% via global in single-session cold run, which
      meets the design intent. Do NOT fix this without real-session evidence that it hurts.

- [x] **M4 extended + Soak test — Round 2**: 4-turn OpenCode session with `bash`, `webfetch`,
      and `Task` sub-agent spawn. Log: `logs/2026-03-13/cache-session-12b90c3add432cba.log`.
      Full analysis in DDR.md Real-Traffic Audit Round 2.

      **Summary:**
      - Cache hit rate: 99.2–99.8% across all 4 turns. ✅
      - All prior messages byte-identical between turns (`diff_turns.py` confirms). ✅
      - Tool results (web fetch content) do NOT appear in history — stripped by OpenCode
        before messages are forwarded. Pattern remains pure tail-append. ✅
      - `Task` sub-agent did not produce a visible second `/v1/chat/completions` call —
        ran inside OpenCode process; result folded back silently. The second log file is
        OpenCode's internal title-generator, not the user-spawned sub-agent. ⚠️
      - `<think>` blocks survive intact in assistant history messages (healing store working). ✅
      - System message stable across all turns (no volatile fields at per-message level). ✅

      **What remains untested:** OpenClaw client, context compaction (long sessions 20+
      turns), scenarios where the client modifies prior assistant messages, real
      mid-history tool-result injection (vs OpenCode's current tail-append pattern).
      See OC_TEST.md for the full repeatable test protocol.

- [x] **Regression check** — confirmed via `probe-normal` probe (2026-03-13):
      `cache_match_type` is `"shorter"` (global path, no secondary lookup) when the global hit
      is already good. The secondary only fires (`stable_prefix_*`) when the global match is
      weaker than the stable prefix estimate. No regression on pure-append sessions.

- [x] **OpenClaw real-traffic audit (Round 1)**: INVALIDATED — Qwen3.5 did not emit
      structured `tool_calls`. Tools appeared to be invoked (model output tool call syntax)
      but `finish_reason=stop` was returned; gateway never dispatched any tool. The file
      `/tmp/oc_claw_test.txt` was never created. All 7 turns were plain conversation.
      Log: `logs/2026-03-13/cache-session-bd7087f6ef7a0c43.log`. Full analysis in DDR.md.

      Valid side-findings from Round 1:
      - [x] Q3: `[Fri 2026-03-13 11:02 GMT+1]` timestamp injected in every user message,
            frozen at send time — not volatile, no FP-1 impact. ✅

      Still unverified (require Round 2 with working tool calls):
      - [ ] Q1: Are `toolResult` messages forwarded verbatim to the LLM?
      - [ ] Q2: Are `toolResult` messages byte-stable across turns?
      - [ ] Q5: Does `sessions_spawn` create a new `/v1/chat/completions` HTTP request?
      - [ ] Q6: Does the stable-prefix secondary lookup fire under real tool-call traffic?
      - [ ] Q7: What is the hit rate when tool results add real content to history?
      - [ ] Q8: Are there new volatile field patterns under real tool execution?

- [ ] **OpenClaw Round 2 — prerequisite: working tool calls** (blocked on model
      tool-calling capability). Three options documented in DDR.md:
      - Option A: Add tool-call extraction layer to MLX server (server-side change)
      - Option B: Use GLM-4.7-Flash via ollama (already configured, known to tool-call)
      - Option C: Inspect existing `pm_Spock` + GLM-4.7-Flash cron session logs
        (real traffic, tools ran, already in `~/.openclaw/agents/pm_spock/sessions/`)

- [x] **OpenClaw M4 extended normalisation**: NOT NEEDED for timestamp injection.
      `[Fri 2026-03-13 11:02 GMT+1]` is frozen at send time, does not cause cache misses.

- [ ] **Context compaction test** (blocked on long session): Run a 20+ turn OpenClaw
      session to trigger context pruning (`contextPruning.mode: cache-ttl`). Observe
      whether the compaction event causes a dramatic cache miss or whether the
      stable-prefix layer recovers correctly from a shortened context.

---

### Known limitations added in Phase 4

**L7 — Heartbeat / side-session collision (mitigated, not fully eliminated):**
When a heartbeat or background agent uses the same implicit session ID as the main conversation
(because the first 128 tokens of both prompts are identical), both entries coexist in the global
cache. After Phase 4 Fix 4, the longest entry always wins in `fetch_nearest_cache`, so the main
conversation's deep cache is no longer displaced by the heartbeat's shallow one. However, the
heartbeat entry still occupies a cache slot and will be returned if the main conversation entry
is evicted. Long-term fix: per-session cache partitioning (already scaffolded in `SessionIndex`
but not yet activated for lookup). Do not activate without real-session evidence it helps.

**L8 — `mx.metal.clear_cache()` called on every `_delete` (including `_cull_redundant_prefixes`):**
Flushing the Metal pool on every single deletion adds a small synchronisation cost. In practice,
`_cull_redundant_prefixes` runs on every insert and may call `_delete` 0–2 times. The cost is
negligible compared to the KV prefill savings, but worth revisiting if insert latency becomes
observable.

---

---

## Phase 5 — Critical Bug: Context Lobotomy via Normalization Regex (2026-03-14)

### Root cause

`INBOUND_TRUSTED_CONTEXT_BLOCK_PATTERN` was defined as:

```python
re.compile(
    r"## Group Chat Context\s*\n"
    r"## Inbound Context \(trusted metadata\)\s*\n"
    r".*?```json\s*\n.*?\n```\s*\n?",
    re.DOTALL,
)
```

The `.*?` before ` ```json ` was intended to skip over any text between the section header
and the JSON code block. When OpenClaw sends the block already pre-normalized
(`__CACHE_STABLE_INBOUND_CONTEXT__` in place of the real JSON), there is no immediate
` ```json ` to match — so `.*?` (with `re.DOTALL`) scans forward across the entire
`# Project Context` section and terminates at the **first ` ```json ``` ` found inside
AGENTS.md or TOOLS.md** (char 16,227 → 34,062: 17,835 chars deleted).

Result: AGENTS.md, SOUL.md, TOOLS.md, USER.md, HEARTBEAT.md, MEMORY.md were silently
stripped from the prompt before tokenization. The model received a 25,731-char prompt
instead of the correct 43,473-char prompt. The agent was blind to its entire identity,
team roster, and operational protocol.

Evidence: `logs/2026-03-14/cache-session-bd7087f6ef7a0c43.log` — system message 43,473
chars in `messages[]`, but rendered prompt fed to model was lobotomized.

### Fix (applied 2026-03-14)

Removed the `.*?` lookahead — pattern now requires the ` ```json ` block to appear
**immediately** after the section header (no intervening content):

```python
re.compile(
    r"## Group Chat Context\s*\n"
    r"## Inbound Context \(trusted metadata\)\s*\n"
    r"```json\s*\n.*?\n```\s*\n?",
    re.DOTALL,
)
```

When the block is already stable (`__CACHE_STABLE_INBOUND_CONTEXT__` present), the
pattern correctly produces no match and the document is left intact.

### Phase 5 follow-up bug (discovered 2026-03-14, fixed same day)

The Phase 5 fix was **too strict**: OpenClaw's real format has 3–4 lines of description
text between `## Inbound Context (trusted metadata)` and the ` ```json ` code fence.
The strict pattern produced **0 matches** on every real OpenClaw request.

As a result, the `elif "# Project Context"` fallback fired on every request, prepending
a synthetic `## Group Chat Context / ## Inbound Context / __CACHE_STABLE_INBOUND_CONTEXT__`
block immediately before `# Project Context`. This left the original real-JSON Inbound
Context block untouched AND added a duplicate stable-token block — two Inbound Context
blocks in the model's context on every turn.

**Fix**: relaxed the pattern to allow up to 10 description lines between the header and
the ` ```json ` fence, anchored by negative lookahead so it cannot cross `## ` or
`# Project Context` boundaries:

```python
re.compile(
    r"## Group Chat Context\s*\n"
    r"## Inbound Context \(trusted metadata\)\s*\n"
    r"(?:(?!## |# Project Context)[^\n]*\n){0,10}"
    r"```json\s*\n.*?\n```\s*\n?",
    re.DOTALL,
)
```

Verified against real log (`logs/2026-03-14/cache-session-bd7087f6ef7a0c43.log`):
- Pre-fix: 2 Inbound Context blocks in normalized output, 44,054 chars
- Post-fix: 1 Inbound Context block, 43,473 chars (correct, matches Phase 5 evidence)

---

## Phase 6 — Dual-Pipeline Normalization Architecture (ACTIVE)

> **Status: active — start here.**
>
> Root cause of every normalization bug in this project: the cache-key computation
> and the model input are the same string, mutated in-place. Any normalization step
> that is correct in 99% of inputs can be catastrophic in the 1% where it
> over-matches. The fix is architectural: **split into two separate pipelines at the
> message-struct level, before rendering.** This is the SOTA pattern used by vLLM,
> SGLang, and Anthropic's own prompt-caching spec.

### Architectural decision (settled — do not re-litigate)

The Phase 5 lobotomy bug and the Phase 5 follow-up duplicate-block bug are both
symptoms of the same root cause: `_normalize_prompt_for_cache()` mutates the rendered
prompt string in-place, and `re.DOTALL` patterns on a flat string can silently consume
user-controlled content. No amount of careful regex tuning permanently fixes this —
every new client format or content variation is a new opportunity for over-matching.

**The correct architecture is the dual pipeline** (see AGENTS.md North Star):

```
[1] _canonicalize_messages(messages)   ← structured data, before rendering
      returns (original_messages, canonical_messages)

[2] apply_chat_template(canonical_messages)  → cache_key_tokens  (lookup only)

[3] apply_chat_template(original_messages)   → model_input_tokens (prefill + generate)

[4] Post-render scrub on cache_key_tokens ONLY
      ← atomic, line-scoped patterns (cch=, billing header, request IDs)
      ← NO re.DOTALL on the full rendered string, ever
```

The model always receives `original_messages` rendered intact. The cache key uses
`canonical_messages`. The two pipelines diverge at step [1] and never recombine.

### Current state defects (to be eliminated)

| Defect | Location | Impact |
|--------|----------|--------|
| In-place mutation of rendered prompt | `_normalize_prompt_for_cache()` called on rendered string | Phase 5 lobotomy, Phase 5 follow-up duplicate block |
| Double normalization on VLM path | `_prepare_messages_for_vlm` L1534 + `_vlm_prompt_and_inputs` L1628 | Second pass sees synthetic tokens from first pass; mismatch risk |
| Model input == cache key | `cache_prompt = _normalize_prompt_for_cache(prompt)` L2561 | Normalization errors affect what the model sees, not just the cache key |
| No safety invariant | None | Silent deletions go undetected until agent reports blindness |
| `elif "# Project Context"` fallback | `_normalize_prompt_for_cache()` L1978 | Fires when pattern doesn't match, prepends synthetic block; caused duplicate-block bug |

### Work steps

#### Step 1 — Introduce `_canonicalize_messages()` (message-struct level)

- [ ] **Write `_canonicalize_messages(messages)`** — takes the raw healed message list,
      returns `(original_messages, canonical_messages)`. Both are deep copies; original
      is never modified. Canonical copy has volatile fields replaced with stable tokens:
      - `message_id` values in JSON content → `__STABLE_MSG_ID__`
      - Inbound metadata JSON blocks (OpenClaw `## Inbound Context`) → `__STABLE_INBOUND_META__`
      - Any other per-message volatile field identified from real logs
      The replacement must happen by **parsing the structured content**, not by regex on
      the rendered string. For the Inbound Context block: locate it in the system message
      `content` string by finding the structural anchor (`## Group Chat Context\n##
      Inbound Context (trusted metadata)\n`), then replace only the JSON payload that
      follows — not anything beyond the closing ` ``` ` fence.
- [ ] **Classify every existing pattern in `_normalize_prompt_for_cache()`** — for each:
      - Can it be moved to `_canonicalize_messages()` (message-level, pre-render)?
      - If not, is it truly atomic and line-scoped (safe for post-render key-only scrub)?
      - If neither: document why and what structural guarantee prevents over-matching.
      Patterns that cannot satisfy either condition must be removed.

#### Step 2 — Wire dual pipeline into `do_POST`

- [ ] **LM path** (non-VLM, line ~2545):
      Replace:
      ```python
      messages = _prepare_messages_for_template(healed_messages)
      prompt = tokenizer.apply_chat_template(messages, ...)
      cache_prompt = _normalize_prompt_for_cache(prompt)
      prompt_tokens = _tokenize_prompt(cache_prompt)
      ```
      With:
      ```python
      original_messages, canonical_messages = _canonicalize_messages(healed_messages)
      messages = _prepare_messages_for_template(original_messages)   # model input
      cache_messages = _prepare_messages_for_template(canonical_messages)  # cache key
      prompt = tokenizer.apply_chat_template(messages, ...)          # model sees this
      cache_prompt = tokenizer.apply_chat_template(cache_messages, ...)
      cache_prompt = _scrub_cache_key(cache_prompt)                  # atomic post-render scrub
      prompt_tokens = _tokenize_prompt(cache_prompt)                 # cache lookup key
      ```
      `SESSION_TURN_STORE` diffs use `messages` (original) — the stable-prefix layer
      already operates correctly on original messages.

- [ ] **VLM path** (line ~2513): same split. Remove `_normalize_prompt_for_cache` call
      from `_prepare_messages_for_vlm` (L1534) and from `_vlm_prompt_and_inputs` (L1628).
      VLM cache key uses canonical messages through the VLM processor; model input uses
      original messages.

- [ ] **Rename / replace `_normalize_prompt_for_cache()`**:
      - Rename to `_scrub_cache_key(text)` — makes its role explicit (cache key only,
        never model input).
      - Remove the `INBOUND_TRUSTED_CONTEXT_BLOCK_PATTERN` substitution and the
        `elif "# Project Context"` fallback entirely — these are now handled at the
        message-struct level in `_canonicalize_messages()`.
      - Keep only atomic, line-scoped patterns: `CACHE_CCH_PATTERN`,
        `CACHE_BILLING_HEADER_PATTERN`, `CACHE_TIME_PATTERN`,
        `CACHE_SYSTEM_REMINDER_PATTERN`. All are single-line or structurally bounded.

#### Step 3 — Safety invariant

- [ ] **Add `_assert_cache_key_safety(original_prompt, cache_key_prompt)`** — callable
      when `CACHE_NORM_SAFETY_CHECK=true` (env var, off by default). Asserts:
      - `len(cache_key_prompt) >= len(original_prompt) * 0.90` — normalization must not
        delete more than 10% of prompt characters. If violated: log an error, fall back
        to using `original_prompt` as the cache key (safe degradation), never crash.

#### Step 4 — Eliminate double normalization

- [ ] **Confirm and remove** the `_normalize_prompt_for_cache(formatted)` call in
      `_vlm_prompt_and_inputs` (line ~1628). After the dual-pipeline refactor, this
      call is redundant (canonical messages were already processed at step [1]).
- [ ] **Confirm and remove** the `_normalize_prompt_for_cache(text)` calls inside
      `_prepare_messages_for_vlm` (lines ~1534, ~1541). Same reason.

#### Step 5 — Regression validation

- [ ] **Regression test against Phase 5 log** — run the raw system message from
      `logs/2026-03-14/cache-session-bd7087f6ef7a0c43.log` through `_canonicalize_messages()`.
      Assert: all 8 workspace files (AGENTS.md, SOUL.md, TOOLS.md, etc.) survive intact
      in `original_messages`. Assert: canonical form has exactly 1 `__STABLE_INBOUND_META__`
      token in place of the JSON payload.
- [ ] **Probe session validation** — run `scripts/probe_session.py` all scenarios.
      Cache hit rates must be unchanged or improved vs Phase 3 baseline.
- [ ] **Log the dual-pipeline split** in per-request telemetry: add
      `cache_key_normalized: bool` and `cache_key_delta_chars: int` fields to the
      prompt log so future sessions can confirm the invariant is holding.

---

### What this phase does NOT change

- `SESSION_TURN_STORE` / stable-prefix layer (Phase 2): operates on `original_messages`,
  unaffected.
- Block-based hash chain (`LRUPromptCache`): operates on `prompt_tokens` (cache key),
  unaffected — it will now receive canonicalized tokens instead of normalized-rendered
  tokens, which is strictly better for cache stability.
- Longest-candidate selection (Phase 4 Fix 4): unaffected.
- Tool-call checkpoint gating (Phase 4 Fix 3): unaffected.

---

*Last updated: 2026-03-14 (Phase 6 rewritten: dual-pipeline architecture — architectural
decision settled, concrete implementation steps, no more research loop)*
