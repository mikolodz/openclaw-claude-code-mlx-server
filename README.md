# MLX LLM Server

A high-performance, local LLM inference server optimised for Apple Silicon via the MLX framework. Exposes an OpenAI-compatible API through a LiteLLM proxy.

## Overview

- **SOTA Caching**: KV-cache-aware inference with a dual-pipeline architecture.
- **Apple Silicon Native**: Built on MLX for maximum performance on macOS.
- **OpenAI Compatible**: Seamless integration with existing tools via a LiteLLM proxy.
- **Reasoning Support**: Native handling of `<think>` blocks with healing and output stripping.
- **Multi-Agent Safe**: Designed for complex workflows where multiple agents share the same model.

## Core Mandates (The "North Star")

This project adheres to strict engineering rules to ensure correctness and performance:

1. **The model must never be blinded**: Semantic content (tool results, `<think>` blocks, file contents) always reaches the model 100% intact. We never hide content to force a cache hit.
2. **Separate the cache key from the model input**: We compute a normalised token sequence for cache lookup ("canonical pipeline"), but always feed the original, unmodified rendered prompt to the model ("original pipeline").
3. **Normalisation operates on structured data**: Volatile fields (IDs, timestamps) are normalised at the message-struct level before rendering. Post-render string manipulation is reserved for atomic, line-scoped patterns only.
4. **RoPE Correctness**: KV state reuse is strictly position-aware. We use physical KV offsets to slice token sequences, ensuring positional encodings are always correct.

## The Dual-Pipeline Architecture

The server uses two parallel pipelines to ensure that cache stability never compromises model reasoning.

```
Incoming Request
      │
      ▼
[1] _canonicalize_messages(messages) ──────────┐
    ├── Operates on message structs            │
    ├── Replaces IDs and Inbound Context       │
    └── Returns: (Original, Canonical)         │
      │                                        │
      ▼                                        ▼
[2] Original Messages                  [3] Canonical Messages
    │   (Model Input Pipeline)             (Cache Key Pipeline)
    │                                          │
    ▼                                          ▼
apply_chat_template()                  apply_chat_template()
    │                                          │
    ▼                                          ▼
model_tokens                           cache_prompt_raw
    │                                          │
    │                                  [4] _scrub_cache_key()
    │                                      └── Atomic, line-scoped scrubs
    │                                          │
    │                                          ▼
    │                                      prompt_tokens (Cache Key)
    │                                          │
    ▼                                          ▼
Prefill & Generate <───────── Match ───────── Lookup
(using KV state)
```

## Cache Architecture

The cache system consists of two independent layers that work together to maximize hits.

### Layer 1: Global Block-Hash Cache (`LRUPromptCache`)

A purely token-based layer that makes no assumptions about session identity or message structure.

- **Hash Chains**: Prompt tokens are split into 16-token blocks. Each block's hash is chained with the prior block's hash (SHA-256).
- **Longest-Candidate Selection**: When multiple entries share the same early block hashes (e.g., a heartbeat vs. a long conversation), the server always picks the deepest matching entry.
- **Cost-Aware Eviction**: Score = `age / (sqrt(length) × log(frequency))`. Protects long conversations and frequently used prefixes while evicting short, old branches.

### Layer 2: Message-Aware Stable-Prefix (`SESSION_TURN_STORE`)

A secondary layer that recovers hits after mid-history mutations (e.g., tool-result injections or early whitespace drift).

- **Structural Diff**: Finds the longest contiguous leading prefix of messages that are identical between turns.
- **Token Boundaries**: Converts the stable message count to an exact token count via cumulative rendering.
- **Secondary Lookup**: If the global layer misses due to a mid-history insertion, the stable-prefix layer forces a lookup up to the change point, preserving the prefix KV state.

### Positional Correctness (RoPE)

When a cache hit occurs in the canonical pipeline, we use `_kv_cache_offset()` to determine the exact number of tokens stored in the physical KV state. We use this physical offset to slice the **original** `model_tokens`. This ensures that even if the canonical and original pipelines differ in token length, the model always receives continuations at the correct RoPE positional encodings.

## Safety Mechanisms

- **`_assert_cache_key_safety`**: An invariant check that ensures normalization never deletes more than 10% of the prompt. If triggered, it falls back to the original prompt for caching to prevent "model blindness."
- **`HEALING_STORE`**: A SHA-256 keyed store that recovers stripped `<think>` reasoning from assistant messages. This prevents cache misses caused by returning stripped text to the model on the next turn.
- **Metal Crash Fix**: Transparently forces PyTorch/torchvision to CPU to prevent command buffer collisions with MLX during massive prefills.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `lmstudio-community/GLM-4.7-Flash-MLX-4bit` | HuggingFace repo or local path |
| `MAX_KV_SIZE` | `196608` | Maximum KV cache size in tokens |
| `KV_BITS` | `off` | KV quantisation bits (`4`, `8`, or `OFF`) |
| `PROMPT_CACHE_MAX_ENTRIES_GLOBAL` | `16` | Max entries in the global cache |
| `CACHE_CANONICALIZE_TOOL_CONTEXT` | `true` | Enable Dual-Pipeline structured normalization |
| `DEFAULT_THINKING` | `true` | Enable reasoning/thinking output by default |

## Implementation Details

- **Python 3.12**: Strict requirement.
- **Threading**: `ThreadingHTTPServer` with `model_lock` for serialised GPU access.
- **VLM Support**: Automatically detects and handles vision-language models via `mlx-vlm`.

## Troubleshooting

- **Cache Misses**: Check `logs/` for `stable_prefix_msg_count`. If 0, an early message (like system prompt) is drifting.
- **OOM Errors**: Lower `PROMPT_CACHE_MAX_ENTRIES_GLOBAL` or enable `KV_BITS=4`.
- **Reasoning Issues**: Ensure `HEALING_STORE` is active (default). It maintains coherence for COT models.
