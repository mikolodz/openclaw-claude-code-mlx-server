# MLX LLM Server

A high-performance, local LLM inference server for Apple Silicon via the MLX framework. Exposes an OpenAI-compatible API through a LiteLLM proxy. The core value proposition is **maximum KV cache hit rate** to minimize prefill latency (~42s per 10k tokens cold on a 27B model).

## Overview

- **KV-Cache-Aware Inference**: Dual-pipeline architecture separates cache stability from model correctness.
- **Apple Silicon Native**: Built on MLX for maximum performance on macOS Metal.
- **OpenAI Compatible**: Seamless integration with existing tools via LiteLLM proxy.
- **Reasoning Support**: Native `<think>` block handling with healing, stripping, and output isolation.
- **Multi-Agent Safe**: Handles complex workflows (OpenClaw, OpenCode) where multiple agents share the same model.
- **VLM Support**: Automatic detection and handling of vision-language models via `mlx-vlm`.

## Quick Start

```bash
# One-command bootstrap (creates .venv, installs deps, starts server)
python3.12 install_and_run.py

# Manual setup
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python start-llm.py

# Verify
curl -s http://127.0.0.1:4000/v1/models
```

**Request path:** Client -> LiteLLM Proxy (port 4000) -> MLX Server (port 8080) -> MLX Model

## Core Mandates

1. **The model must never be blinded.** Semantic content (tool results, `<think>` blocks, file contents) always reaches the model 100% intact. A cache hit that hides content is worse than a miss.
2. **Separate the cache key from the model input.** Canonical pipeline for cache lookup, original pipeline for model. The two never recombine after `_canonicalize_messages()`.
3. **Normalisation operates on structured data.** Volatile fields are normalised at message-struct level before rendering. Post-render string manipulation is atomic, line-scoped only. No `re.DOTALL` spanning section boundaries.
4. **RoPE Correctness.** KV state reuse is strictly position-aware. Physical KV offsets (`cache[0].offset`) are used to slice token sequences, never canonical token counts.

## Architecture

### The Dual Pipeline

The server runs two parallel pipelines to ensure cache stability never compromises model reasoning.

```
Incoming Request
      |
      v
[1] _heal_messages()
      |  Restores stripped <think> blocks from HEALING_STORE
      v
[2] _canonicalize_messages(messages)
    |-- Returns: (Original, Canonical)
    |-- Replaces: message_id, Inbound Context block, sub-agent stats
      |                                        |
      v                                        v
[3] Original Messages                  [4] Canonical Messages
    (Model Input Pipeline)                 (Cache Key Pipeline)
      |                                        |
      v                                        v
  apply_chat_template()               apply_chat_template()
      |                                        |
      v                                        v
  model_tokens                         cache_prompt_raw
      |                                        |
      |                                [5] _scrub_cache_key()
      |                                    Atomic, line-scoped scrubs:
      |                                    - timestamps
      |                                    - cch= headers
      |                                    - billing headers
      |                                    - <system-reminder> blocks
      |                                        |
      v                                        v
  Prefill & Generate  <--- KV Match --- prompt_tokens (Cache Key)
  (using KV state)                      Lookup via LRUPromptCache
```

### Canonicalization Layers

| Layer | Function | Scope | What it normalises |
|---|---|---|---|
| Message-struct | `_canonicalize_messages()` | Before rendering | `message_id` values, Inbound Context block, sub-agent stats |
| Inbound Context | `_canonicalize_inbound_context_block()` | Inside `_canonicalize_messages` | Produces identical output whether OpenClaw includes or omits the block |
| Post-render | `_scrub_cache_key()` | After `apply_chat_template` | Timestamps, `cch=` headers, billing headers, `<system-reminder>` tags |

## Cache System

The cache consists of two independent layers that work together to maximize hits.

### Layer 1: Global Block-Hash Cache (`LRUPromptCache`)

A token-based layer that makes no assumptions about session identity or message structure.

- **Hash Chains**: Prompt tokens are split into 16-token blocks. Each block's hash is chained with the prior block's hash (SHA-256).
- **Longest-Candidate Selection**: When multiple entries share the same early block hashes (e.g., a heartbeat vs. a long conversation), the server always picks the deepest matching entry. Without this, a short entry can shadow a long one and produce 2.6% hit instead of 93%.
- **Cost-Aware Eviction**: Score = `age / (sqrt(length) * log(frequency))`. Protects long conversations and frequently used prefixes while evicting short, old branches.
- **Prefix Culling**: When a longer entry is inserted, shorter strict prefixes are removed (keeping the longest prefix as a safety net).
- **Metal Memory Management**: `mx.metal.clear_cache()` is called on eviction to prevent monotonic GPU memory growth.

### Layer 2: Message-Aware Stable-Prefix (`SESSION_TURN_STORE`)

A secondary layer that recovers hits after mid-history mutations (e.g., tool-result injections or early whitespace drift).

- **Structural Diff**: `_message_diff()` finds the longest contiguous leading prefix of messages identical between turns.
- **Token Boundaries**: `_compute_msg_token_boundaries()` converts the stable message count to an exact token count via cumulative `apply_chat_template(messages[:i+1])` rendering.
- **Secondary Lookup**: If the global layer returns fewer cached tokens than the stable-prefix layer estimates, a secondary lookup forces a match up to the change point, preserving prefix KV state.
- **Write Guard**: Only records strict appends of the prior message list. Prevents sub-agent or parallel-branch requests from clobbering the orchestrator's turn record.

### Positional Correctness (RoPE)

When a cache hit occurs, `_kv_cache_offset(cache)` reads the physical KV offset from `cache[0].offset`. This offset is used to slice the **original** `model_tokens` for the remaining prefill. This ensures correct RoPE positional encodings even when canonical and original pipelines differ in token length (e.g., a 200-token JSON block becomes a 4-token `__STABLE_INBOUND_META__` sentinel in the canonical pipeline).

### Session Management

- **`SessionIndex`**: Tracks per-session cache key history with a lineage chain (parent/branch relationships). Supports anchor prefixes at 2048-token strides for branch-return reuse.
- **`SessionContext`**: Extracted from request body fields (`session_id`, `conversation_id`, `thread_id`, etc.) or derived from prompt prefix hash when no explicit ID is provided.

## Reasoning and Healing

### `<think>` Block Handling

The server supports chain-of-thought reasoning models (Qwen3, GLM-4) with `<think>` blocks:

1. **Generation**: Model generates with `enable_thinking=True` (configurable per-request via `enable_thinking`, `thinking`, `reasoning_effort`, or `reasoning` fields).
2. **Client Output**: `<think>` blocks are stripped before returning to the client, so reasoning is hidden.
3. **Healing**: When the client sends back the stripped assistant message on the next turn, the `HEALING_STORE` (SHA-256 keyed) restores the full text including `<think>` blocks. This prevents cache misses from stripped-then-re-sent reasoning text.

### Tool Call Extraction

The server extracts structured OpenAI tool calls from model-generated XML:
- Legacy format: `<tool_call>name <arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
- Qwen format: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`

Extracted tool calls are returned as standard OpenAI `tool_calls` objects in the response.

## Safety Mechanisms

- **`_assert_cache_key_safety`**: Optional invariant check (`CACHE_NORM_SAFETY_CHECK=true`) ensuring normalization never deletes more than 10% of the prompt. Falls back to original prompt as cache key on violation.
- **Metal Crash Fix**: PyTorch MPS is disabled at import time (`torch.backends.mps.is_built = lambda: False`) to prevent Metal command buffer collisions between PyTorch/torchvision and MLX during large prefills.
- **Thread Safety**: `model_lock` serialises GPU access, `prompt_cache_lock` protects all shared cache state, `console_lock` serialises terminal output.

## Environment Variables

### Core

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `lmstudio-community/GLM-4.7-Flash-MLX-4bit` | HuggingFace repo or local path |
| `MODEL_FAMILY` | Auto-detected from path | `qwen3`, `glm4`, or `generic` |
| `PROXY_MODEL_ID` | `openai/{MODEL_PATH}` | Model name exposed through LiteLLM proxy |
| `MLX_HOST` | `127.0.0.1` | MLX server bind address |
| `MLX_PORT` | `8080` | MLX server port |
| `PROXY_PORT` | `4000` | LiteLLM proxy port |
| `PROXY_STARTUP_WAIT_SECONDS` | `2.0` | Seconds to wait for proxy startup |

### KV Cache

| Variable | Default | Description |
|---|---|---|
| `MAX_KV_SIZE` | `196608` | Maximum KV cache size in tokens |
| `KV_GROUP_SIZE` | `64` | KV cache group size |
| `KV_BITS` | `OFF` | KV quantisation bits (`4`, `8`, or `OFF`) |

### Prompt Cache

| Variable | Default | Description |
|---|---|---|
| `PROMPT_CACHE_MAX_ENTRIES_GLOBAL` | `24` | Max entries in the global block-hash cache |
| `PROMPT_CACHE_MAX_ENTRIES_PER_SESSION` | `2` | Max cache keys tracked per session |
| `PROMPT_CACHE_TTL_SECONDS` | `1800` | Time-to-live for cache entries (0 = infinite) |
| `PROMPT_CACHE_SESSION_MAX_IDLE_SECONDS` | `1800` | Session idle timeout for turn store (0 = infinite) |
| `PROMPT_CACHE_BLOCK_SIZE` | `16` | Token block size for hash chains |
| `CACHE_USE_BLOCK_INDEX` | `true` | Enable block-hash index for O(blocks) lookup |
| `CACHE_CANONICALIZE_TOOL_CONTEXT` | `true` | Enable dual-pipeline structured normalisation |
| `CACHE_SESSION_PARTITIONING` | `true` | Session partitioning (ignored; cache is always global) |
| `CACHE_NORM_SAFETY_CHECK` | `false` | Enable the 10%-delta safety invariant check |

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
| `DEFAULT_THINKING` | `true` | Enable reasoning/thinking output by default |

### Logging and Debug

| Variable | Default | Description |
|---|---|---|
| `ENABLE_REQUEST_LOGGING` | `true` | Write per-request session logs to `logs/` |
| `LOG_ROOT` | `logs` | Log output directory |
| `VLM_CACHE_DEBUG` | `false` | Log first 64 prompt token IDs for VLM cache debugging |

## Project Structure

| File | Purpose |
|---|---|
| `start-llm.py` | Entire server (~3500 LOC): API handler, prompt cache, sessions, dual pipeline |
| `install_and_run.py` | Bootstrap: creates `.venv`, installs deps, execs server |
| `.env` / `.env.example` | Runtime config (model path, cache sizes, ports, sampler presets) |
| `requirements.txt` | Dependencies: `mlx`, `mlx-lm`, `mlx-vlm`, `litellm[proxy]`, `python-dotenv` |
| `scripts/diff_turns.py` | Diff consecutive turns from cache session logs |
| `scripts/inspect_mlx_cache.py` | Inspect MLX KV cache state |
| `scripts/probe_session.py` | 4-scenario cache validation (normal, drift, semantic, insert) |
| `scripts/test_openclaw_integration.py` | Automated OpenClaw integration test harness |
| `.context/LESSONS.md` | Hard-won knowledge from Phases 0-8 (read before touching cache logic) |
| `.context/TASKS.md` | Active tasks and open items |
| `.context/GOALS.md` | Strategic direction |

## VLM (Vision-Language Model) Support

The server automatically detects VLM models from `config.json` (checking `model_type`, `vision_config`, and architecture names). When a VLM is loaded:

- Messages with `image_url` or `input_image` content parts are processed via `mlx-vlm`'s `prepare_inputs`.
- Images are extracted from data URLs (base64 decoded to PIL) or passed as URL strings.
- The dual pipeline applies: canonical cache key uses text-only tokenization (CPU, no GPU lock), while the model prefills from the full VLM input including vision tokens.
- Metal sync (`torch.mps.synchronize()` + `mx.eval()`) runs before generation to prevent Metal encoder collisions.

## Troubleshooting

- **Cache Misses**: Check `logs/` for `stable_prefix_msg_count`. If 0, an early message (like system prompt) is drifting. Look at `cache_key_delta_chars` to verify normalisation is active.
- **OOM Errors**: Lower `PROMPT_CACHE_MAX_ENTRIES_GLOBAL` or enable `KV_BITS=4`. At 20k tokens, each cache entry is ~2-3 GB of GPU memory.
- **Reasoning Issues**: Ensure `HEALING_STORE` is active (default). It maintains coherence for COT models across turns.
- **VLM Metal Crash**: The server disables PyTorch MPS at startup. If crashes persist, install `mlx-vlm` without the torch extra: `pip uninstall torch torchvision; pip install mlx-vlm`.
- **Stale LiteLLM**: The server auto-detects and kills stale LiteLLM processes on the proxy port at startup.

## Requirements

- Python 3.12 (strict)
- macOS with Apple Silicon (M1/M2/M3/M4)
- Dependencies: `mlx`, `mlx-lm`, `mlx-vlm` (optional for VLM), `litellm[proxy]`, `python-dotenv`
