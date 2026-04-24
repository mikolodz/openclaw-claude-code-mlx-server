# MLX LLM Server

A local, KV-cache-aware LLM inference server for Apple Silicon built on the MLX framework. Exposes an OpenAI-compatible API through a LiteLLM proxy. The core value proposition is **maximum KV cache hit rate** to minimize prefill latency — on a 27B model, cold 10k-token prefill costs ~42 s while a warm hit is effectively free.

## Overview

- **Dual-pipeline cache architecture**: canonical pipeline drives the cache key, original pipeline drives the model. They never recombine after `_canonicalize_messages()`, so cache stability never compromises model input.
- **Two cache layers**: a global block-hash cache (token-level) and a message-aware stable-prefix cache (structure-level). Secondary lookup recovers hits after mid-history tool-result injections.
- **Apple Silicon native**: MLX + `mlx-lm` for text models, `mlx-vlm` for vision-language models (auto-detected from `config.json`).
- **OpenAI-compatible streaming**: real token-by-token SSE with proper `reasoning_content` and `content` deltas so clients (PI, OpenClaw, Claude Code, LiteLLM) render thinking blocks and answers as they arrive.
- **Multi-agent safe**: canonical-form write guards on the session turn store prevent sub-agent branches from clobbering the orchestrator's turn record; per-session lineage tracking supports branch-return cache reuse.
- **Request watchdog**: wall-clock deadline per request releases `model_lock` on stalls, evicts partial KV state, and terminates the response with `finish_reason="length"`.
- **Correctness observability**: counters (`vlm_retreat`, `hybrid_trim_miss`, `exact_key_rejected_by_model_lcp`) surface cache-correctness regressions as they happen.

## Quick Start

```bash
# One-command bootstrap (creates .venv, installs deps, execs server)
python3.12 install_and_run.py

# Manual setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python start-llm.py

# Smoke test
curl -s http://127.0.0.1:4000/v1/models
```

`install_and_run.py` has a preflight that refuses to run if the venv has been contaminated by a non-3.12 Python (e.g. a stray `python3.14 -m pip` against the existing venv). Python 3.12 is strictly required.

**Request path:** Client -> LiteLLM Proxy (port 4000) -> MLX Server (port 8080) -> MLX Model.

## Core Mandates

1. **The model must never be blinded.** All semantic content (tool results, `<think>` blocks, file contents, images) reaches the model 100% intact. A cache hit that hides content is worse than a miss.
2. **Separate cache key from model input.** Canonical tokens drive cache lookup; original tokens drive the model. The two are computed at `_canonicalize_messages()` and never recombined.
3. **Structural normalisation, not string substitution.** Volatile fields are normalised at message-struct level before `apply_chat_template`. Post-render scrubs are atomic, line-scoped only — no `re.DOTALL` spanning section boundaries.
4. **RoPE correctness.** KV state reuse is position-aware. Physical KV offsets (`_kv_cache_offset(cache)`) slice the **original** `model_tokens` for the remaining prefill; canonical token counts are never used for model-side slicing.
5. **No gut-feel patches.** Every change to caching or normalisation cites a specific log-confirmed failure mode. See `.context/LESSONS.md` before touching cache logic.

## Architecture

### The dual pipeline

```
Incoming Request
      |
      v
[1] _heal_messages()
      |  Restores stripped <think> blocks from HEALING_STORE (SHA-256 keyed)
      v
[2] _canonicalize_messages(messages)
    |-- Returns: (original, canonical)  —  deep copies; original is never modified
    |-- Canonical mutations: message_id -> __STABLE_MSG_ID__,
    |   Inbound Context block -> __STABLE_INBOUND_CONTEXT_SECTION__,
    |   sub-agent "Stats: runtime ..." -> __STABLE_RUNTIME__
      |                                        |
      v                                        v
[3] original_messages                   [4] canonical_messages
    (Model Input Pipeline)                 (Cache Key Pipeline)
      |                                        |
      v                                        v
  apply_chat_template()               apply_chat_template()
      |                                        |
      v                                        v
  model_tokens                        cache_prompt_raw
      |                                        |
      |                               [5] _scrub_cache_key()
      |                                   Atomic, line-scoped scrubs:
      |                                   - "Current time is ..." timestamps
      |                                   - cch=<hex>; telemetry headers
      |                                   - -anthropic-billing-header: ...
      |                                   - <system-reminder>...</system-reminder>
      |                                        |
      |                               [5a] Image-identity markers (G1)
      |                                   For VLM requests: inject a 2-token
      |                                   SHA-256-derived marker after each
      |                                   <|image_pad|> run so identical
      |                                   message structure with different
      |                                   images yields different cache keys
      |                                        |
      v                                        v
  Prefill & Generate  <--- KV Match --- prompt_tokens (canonical cache key)
  (from model_tokens)                   Lookup via LRUPromptCache +
                                         SessionIndex + SESSION_TURN_STORE
```

### Canonicalisation layers

| Layer | Function | Scope | What it normalises |
|---|---|---|---|
| Message-struct | `_canonicalize_messages()` | Before rendering | `message_id`, Inbound Context block, sub-agent `Stats: runtime` |
| Inbound Context | `_canonicalize_inbound_context_block()` | Inside `_canonicalize_messages` | Produces identical canonical output whether OpenClaw includes or omits the block |
| Post-render | `_scrub_cache_key()` | After `apply_chat_template` | Timestamps, `cch=` headers, billing headers, `<system-reminder>` tags |
| Image identity | `_inject_image_markers()` | Post-tokenisation, canonical only | 2-token SHA-256 marker per image — forces cache-key divergence on image swap |

## Cache system

Two independent layers work together to maximise hits.

### Layer 1 — Global block-hash cache (`LRUPromptCache`)

Token-based, no assumptions about session identity or message structure.

- **Chain hashes**: prompt tokens are split into 16-token blocks; each block's hash is SHA-256-chained with the prior block's hash.
- **Longest-candidate selection**: among entries sharing the same early block hashes, always pick the deepest matching entry. Without this, a short entry shadows a long one and collapses hit rate.
- **Model-space LCP verification**: candidate discovery is canonical-token-based, but final acceptance requires `entry.model_tokens` to be a prefix of the request's `model_tokens`. This guards against canonical-prefix collisions across different VLM image expansions.
- **Hybrid-cache awareness**: Qwen3.6 and other hybrid VLMs expose `ArraysCache + KVCache` layers that `can_trim_prompt_cache` reports as non-trimmable. Candidates that would require trim are rejected rather than silently reused at the wrong KV depth.
- **Cost-aware eviction**: score = `age / (sqrt(length) * log(frequency))`. Protects long conversations and frequently-used prefixes; evicts short, old branches.
- **Prefix culling**: when a longer entry is inserted, shorter strict prefixes are removed (longest one kept as a safety net). Requires model-token-space prefix match, not just canonical.
- **Metal memory management**: `mx.metal.clear_cache()` runs on eviction to return Metal buffers to the OS pool and prevent monotonic GPU memory growth.

### Layer 2 — Message-aware stable-prefix (`SESSION_TURN_STORE`)

Secondary layer that recovers hits after mid-history mutations (tool-result injections, early whitespace drift, `message_id` churn).

- **Canonical-form storage and diff**: the turn record stores `_canonicalize_messages()` output, and `_message_diff()` compares canonical-form leading prefixes. This lets OpenClaw drift (per-turn `message_id`, Stats runtime, Inbound Context variants) collapse to equality because it IS equal in cache-key space.
- **Exact token boundaries**: `_compute_msg_token_boundaries()` converts the stable message count to an exact token count via cumulative `apply_chat_template(messages[:i+1])` rendering. Falls back to even distribution only for VLMs without a usable text template.
- **Secondary lookup**: if the global layer returns fewer cached tokens than the stable-prefix layer estimates, a secondary `fetch_nearest_cache` scoped to the stable prefix forces a match up to the change point.
- **Strict-append write guard**: only records when the new canonical list strictly extends the stored one. Prevents sub-agent or parallel-branch requests from clobbering the orchestrator's turn record.

### Positional correctness (RoPE)

On every cache hit, `_kv_cache_offset(cache)` reads the physical KV depth from the first attention layer that exposes `.offset` (hybrid VLM caches mix `ArraysCache + KVCache` — layer 0 may lack it). That offset slices the **original** `model_tokens` for the remaining prefill. RoPE stays coherent even when canonical and original pipelines differ by thousands of tokens (e.g., a 200-token JSON Inbound Context block collapses to a single `__STABLE_INBOUND_CONTEXT_SECTION__` sentinel in the canonical pipeline).

### Session management

- **`SessionIndex`**: per-session cache-key history with parent/branch lineage. Anchor prefixes at 2048-token strides support branch-return reuse.
- **`SessionContext`**: extracted from request body (`session_id`, `conversation_id`, `thread_id`, `chat_id`, or nested `metadata` / `extra_body`). Falls back to a SHA-1 hash of the prompt's first 128 tokens when no explicit ID is provided.

## Reasoning and streaming

### Streaming split: `reasoning_content` vs `content`

The server uses a character-level state machine (`_ReasoningStreamSplitter`) to classify every generated chunk into one of three streams — **reasoning** (to `delta.reasoning_content`), **content** (to `delta.content`), or **tool-call buffer** (held for end-of-stream extraction). The classifier:

- Handles `<think>...</think>` blocks emitted by Qwen3 and the orphan-close variant emitted by GLM ("reasoning</think>\nanswer" with no opening tag).
- Holds a small lookback so tags split across two generated chunks are detected correctly — no premature emission of partial sentinels.
- Matches `_strip_thinking_from_content` byte-for-byte on the concatenated `content` stream for every non-pathological case (verified by import-time self-test plus a 150-trial random-chunking soak). This means the stream a client receives is identical to the non-streaming `message.content`, keeping the healing-hash consistent across `stream=true` / `stream=false` retries.
- Switches to a buffered tool-call mode on the first `<tool_call>` sentinel; end-of-stream extraction then emits `delta.tool_calls` plus any trailing non-tool text.

PI, LiteLLM's OpenAI-compatible wrapper, and other clients that understand `reasoning_content` render thinking live. Clients that don't understand the field see identical `content` and ignore the rest — no back-compat break.

### Non-streaming responses

The non-streaming path still derives `message.content` from the legacy `_strip_thinking_from_content` + `_extract_openai_tool_calls` pipeline (byte-identical to prior versions). When thinking was requested, `message.reasoning_content` is populated additively via the same splitter.

### Healing store

`HEALING_STORE` is a SHA-256-keyed OrderedDict (cap 2000). When a client echoes a stripped assistant message back in the next turn, `_heal_messages` looks up the hash of `(content, canonicalised tool_calls)` and swaps the stripped message for the full original (with `<think>` blocks intact). Tool-call arguments are canonicalised to parsed-dict form before hashing so clients that deserialise `arguments` differently from the server still hit the store.

### Tool calls

The server extracts structured OpenAI `tool_calls` from model-generated XML:

- Legacy: `<tool_call>name <arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
- Qwen: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`

Both patterns are recognised independent of `model_family`; parser order is family-prioritised.

### Reasoning control

Any of these request fields map to `enable_thinking` (bool) internally:

- `enable_thinking: true/false`
- `thinking: off|on|low|medium|high|xhigh|...` (OpenClaw / Anthropic style)
- `reasoning_effort: off|low|medium|high|xhigh` (OpenAI / LiteLLM style)
- `reasoning: {enabled|effort|level|type: ...}` (Responses-style)
- Nested `metadata.*` / `extra_body.*` variants of the above

Default is controlled by `DEFAULT_THINKING` (default `true`).

## VLM (vision-language model) support

VLMs are auto-detected from `config.json` via a whitelist of `mlx-vlm` `model_type` values (Qwen3.6, Qwen2.5-VL, GLM-4V, Llava, Gemma3, etc.). When a VLM is loaded:

- Messages with `image_url` or `input_image` content parts are processed via `mlx_vlm.utils.prepare_inputs`. Data URLs are base64-decoded to PIL; bare URLs pass through as strings.
- The canonical cache key is derived using the **text-only tokeniser** (CPU, no GPU lock needed) so we don't re-acquire `model_lock` for cache lookup.
- **Image identity (G1)**: each image payload yields a 2-token SHA-256-derived marker injected after its `<|image_pad|>` run in the canonical token stream. Two requests with the same message structure but different images now produce different cache keys. Without this, `<|image_pad|>` is a single token id that encodes identically for any image, and the block-hash index happily serves image A's KV for a request carrying image B.
- **Partial-image safety net**: if a cache cut lands mid-image-pad-run (diagnosed via `_vlm_cache_covers_partial_image`), the server evicts the offending entry, retreats to fresh prefill, and bumps the `vlm_retreat` counter. With G1 active, this should never fire on clean traffic — any bump is a signal.
- Metal sync (`torch.mps.synchronize()` + `mx.eval()`) runs before generation to prevent Metal encoder collisions between PyTorch/torchvision and MLX.

## Safety and correctness

### Generation watchdog (G2)

Every request arms a `threading.Timer(GENERATION_WATCHDOG_SECONDS)` after acquiring `model_lock`. If it fires, a cooperative abort flag is set; the generator yield loop checks it between tokens and exits cleanly. On abort: partial KV state is **not** inserted into the cache, the session turn record is **not** advanced, the healing store is **not** updated, and the response terminates with `finish_reason="length"`. The timer is cancelled unconditionally in the `finally` block so `model_lock` always releases. Default 600 s — set `GENERATION_WATCHDOG_SECONDS=0` to disable.

### Canonical-form write guard (G3)

`_update_session_turn_store` canonicalises the incoming message list and compares its leading prefix against the stored canonical form. Only strict appends in canonical space advance the record; any semantic divergence (new system prompt, mid-history edit, sub-agent branch) refuses the write to preserve the best-known linear record.

### Cache-correctness counters (P4)

Three counters, atomically bumped and surfaced per-request in logs and at startup:

| Counter | Meaning | Expected value |
|---|---|---|
| `vlm_retreat` | Mid-image cache cut forced a fresh prefill | 0 on clean traffic (G1 effectiveness) |
| `hybrid_trim_miss` | Hybrid VLM cache rejected because `can_trim_prompt_cache` is False | 0 or low; justifies G4 (pre-generation checkpoint) |
| `exact_key_rejected_by_model_lcp` | Exact canonical key hit rejected because `model_tokens` diverged | 0 — any bump is a canonicalisation-over-normalisation bug |

### Thread safety

- `model_lock` — serialises GPU access (prefill + decode).
- `prompt_cache_lock` — protects `PROMPT_CACHE`, `SESSION_TURN_STORE`, `SESSION_INDEX`.
- `HEALING_STORE_LOCK` — protects the healing OrderedDict.
- `console_lock` — serialises terminal output.
- `_CACHE_METRICS_LOCK` — protects the correctness counter dict.

### HTTP server fairness

The server runs single-threaded (`HTTPServer`) because MLX streams are per-thread and `mlx-vlm`'s `generate_step` issues bare `mx.async_eval` outside a `with mx.stream(...)` context. Every response sets `Connection: close` and `self.close_connection = True` so the accept loop doesn't park on a keep-alive socket from LiteLLM's connection pool. When upstream mlx-vlm moves to `mx.new_thread_local_stream`, this can revert to `ThreadingHTTPServer`.

### Other

- **`_assert_cache_key_safety`** (opt-in via `CACHE_NORM_SAFETY_CHECK=true`): asserts the cache key is not dramatically shorter than the original prompt (>10% delta = over-match). Falls back to original prompt as cache key on violation.
- **Metal crash fix**: PyTorch MPS disabled at import time (`torch.backends.mps.is_built = lambda: False`) to prevent Metal command buffer collisions with MLX.
- **Preserve-thinking probe**: at startup, `_probe_preserve_thinking_support()` tests whether the active tokenizer/processor accepts the `preserve_thinking` kwarg, caching the result so `apply_chat_template` calls stay consistent across cache-key and model-input paths.

## Environment variables

### Core

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `lmstudio-community/GLM-4.7-Flash-MLX-4bit` | HuggingFace repo or local path |
| `MODEL_FAMILY` | auto-detected from path | `qwen3`, `glm4`, or `generic` |
| `PROXY_MODEL_ID` | `openai/{MODEL_PATH}` | Model name exposed through LiteLLM |
| `MLX_HOST` | `127.0.0.1` | MLX server bind address |
| `MLX_PORT` | `8080` | MLX server port |
| `PROXY_PORT` | `4000` | LiteLLM proxy port |
| `PROXY_STARTUP_WAIT_SECONDS` | `2.0` | Seconds to wait for LiteLLM startup |

### KV cache

| Variable | Default | Description |
|---|---|---|
| `MAX_KV_SIZE` | `196608` | Maximum KV cache size in tokens |
| `KV_GROUP_SIZE` | `64` | KV cache group size (uniform scheme) |
| `KV_BITS` | `OFF` | KV quantisation bits (`4`, `8`, or `OFF`) |
| `KV_QUANT_SCHEME` | `uniform` | `uniform` or `turboquant` |
| `QUANTIZED_KV_START` | `5000` | Token position where KV quantisation kicks in |

### Prompt cache

| Variable | Default | Description |
|---|---|---|
| `PROMPT_CACHE_MAX_ENTRIES_GLOBAL` | `24` | Max entries in the global block-hash cache |
| `PROMPT_CACHE_MAX_ENTRIES_PER_SESSION` | `2` | Max cache keys tracked per session |
| `PROMPT_CACHE_TTL_SECONDS` | `1800` | Time-to-live for cache entries (0 = infinite) |
| `PROMPT_CACHE_SESSION_MAX_IDLE_SECONDS` | `1800` | Session idle timeout for turn store (0 = infinite) |
| `PROMPT_CACHE_BLOCK_SIZE` | `16` | Token block size for hash chains |
| `CACHE_USE_BLOCK_INDEX` | `true` | Enable block-hash index for O(blocks) lookup |
| `CACHE_CANONICALIZE_TOOL_CONTEXT` | `true` | Enable dual-pipeline structured normalisation |
| `CACHE_VLM_IMAGE_IDENTITY` | `true` | Inject per-image markers into canonical cache key (G1) |
| `CACHE_SESSION_PARTITIONING` | `true` | Session partitioning flag (read but not applied — lookup is always global) |
| `CACHE_NORM_SAFETY_CHECK` | `false` | Enable the 10%-delta cache-key safety assertion |
| `NORMALIZE_WRITE_TOOL_CONTENT_FOR_PROMPT` | `false` | Replace `write` tool's `content` arg with a digest placeholder at render time |

### Generation

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_TEMPERATURE` | `0.1` | Sampling temperature |
| `DEFAULT_TOP_P` | `0.9` | Top-p (nucleus) sampling |
| `DEFAULT_TOP_K` | `40` | Top-k sampling |
| `DEFAULT_MIN_P` | `0.05` | Minimum probability threshold |
| `DEFAULT_REPETITION_PENALTY` | `1.08` | Repetition penalty |
| `DEFAULT_REPETITION_CONTEXT_SIZE` | `256` | Context window for repetition penalty |
| `DEFAULT_MAX_TOKENS` | `2048` | Default max output tokens |
| `DEFAULT_THINKING` | `true` | Enable reasoning output by default |
| `PRESERVE_THINKING` | `true` | Forward `preserve_thinking=True` to `apply_chat_template` when the tokeniser supports it |
| `GENERATION_WATCHDOG_SECONDS` | `600` | Wall-clock deadline per request (0 = disabled) |

### Logging and debug

| Variable | Default | Description |
|---|---|---|
| `ENABLE_REQUEST_LOGGING` | `true` | Write per-request session logs to `logs/` |
| `LOG_ROOT` | `logs` | Log output directory |
| `VLM_CACHE_DEBUG` | `false` | Log first 64 prompt token IDs and cache-diagnostic lines for VLM debugging |

## Project structure

| File | Purpose |
|---|---|
| `start-llm.py` | Entire server (~4900 LOC): API handler, dual pipeline, prompt cache, session layers, streaming splitter, watchdog |
| `install_and_run.py` | Bootstrap: verifies Python 3.12, detects venv contamination, installs deps, execs server |
| `.env` / `.env.example` | Runtime config |
| `requirements.txt` | Dependencies: `mlx`, `mlx-lm`, `mlx-vlm`, `litellm[proxy]`, `python-dotenv` |
| `scripts/pi_like_probe.py` | Primary cache-hit / divergence validation harness — PI-shaped agentic probe |
| `scripts/probe_session.py` | 4-scenario cache validation (normal, drift, semantic, insert) |
| `scripts/diff_turns.py` | Diff consecutive turns from cache session logs |
| `scripts/inspect_mlx_cache.py` | Inspect MLX KV cache state |
| `scripts/test_openclaw_integration.py` | Automated OpenClaw integration test harness |
| `.context/GOALS.md` | Strategic direction, active G-items (G1–G9) |
| `.context/TASKS.md` | Active portions and open items |
| `.context/LESSONS.md` | Hard-won knowledge — read before touching cache logic |
| `.claude/rules/server-core.md` | Auto-loaded rules when editing `start-llm.py` |
| `CLAUDE.md` | Project conventions and change protocol |

## Supported models

Any MLX-compatible text LM and any `mlx-vlm` 0.4.x-supported VLM. Tested and actively tuned for:

- **Qwen3 / Qwen3.6** (text and hybrid VLM) — primary model_family
- **GLM-4 / GLM-4.7-Flash** — primary model_family with orphan-close `</think>` handling
- Generic LMs via mlx-lm's `load()` path

Hybrid VLMs (Qwen3.6's `Qwen3_5MoeForConditionalGeneration`, mixing `ArraysCache` and `KVCache` layers) are handled with explicit offset scanning — layer 0 may lack `.offset`, so `_kv_cache_offset` walks layers to find the first attention cache.

## Verification

No formal unit-test suite; verification is empirical via real agentic traffic.

- `python scripts/pi_like_probe.py` — PI-shaped agentic probe, primary harness.
- `python scripts/probe_session.py` — 4-scenario cache validation.
- `scripts/diff_turns.py` — diff session prompts turn-by-turn to pinpoint cache-key divergence.
- `logs/<date>/cache-session-*.log` — per-session cache trace; first place to look when a fix may have regressed hit rate.

Healthy numbers: text T3 ≥ 97% hit, image T2 ≥ 94% hit, zero bumps on `vlm_retreat` and `exact_key_rejected_by_model_lcp`, `hybrid_trim_miss` small and only on cross-session cold transitions.

## Troubleshooting

- **Cache misses**: inspect `logs/` for `stable_prefix_msg_count`. If 0, an early message is drifting — check `cache_key_delta_chars` to verify canonicalisation fired. If `exact_key_rejected_by_model_lcp` is bumping, the canonical pipeline is over-normalising.
- **Server hangs**: watchdog should catch after `GENERATION_WATCHDOG_SECONDS`. If `model_lock` is stuck, lower the timeout or investigate the Metal state via `Activity Monitor`.
- **OOM**: lower `PROMPT_CACHE_MAX_ENTRIES_GLOBAL` or enable `KV_BITS=4`. At 20k tokens each cache entry is ~2–3 GB of GPU memory.
- **Reasoning not rendering in client**: verify client reads `delta.reasoning_content` (OpenAI Qwen/DeepSeek convention). Clients that only read `delta.content` see the answer without the thinking block — this is expected.
- **VLM Metal crash**: the server disables PyTorch MPS at startup. If crashes persist, install `mlx-vlm` without the torch extra: `pip uninstall torch torchvision; pip install mlx-vlm`.
- **Stale LiteLLM**: auto-detected and killed on the proxy port at startup.
- **Venv contamination** (e.g. `python3.14` binaries inside `.venv/`): `install_and_run.py` refuses to proceed and points at the artefacts. Delete the venv and re-run, or scrub the bin entries and restore `pip -> pip3.12` symlinks.

## Requirements

- **Python 3.12 strict** — enforced by `install_and_run.py` preflight.
- macOS with Apple Silicon (M1/M2/M3/M4).
- Dependencies: `mlx`, `mlx-lm`, `mlx-vlm` (optional — only loaded for VLM models), `litellm[proxy]`, `python-dotenv`.
