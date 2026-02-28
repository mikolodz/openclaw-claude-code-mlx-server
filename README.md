# MLX LLM Module Documentation

A high-performance, local LLM inference server using Apple's MLX framework with advanced caching, multi-modal (VLM) support, and OpenAI-compatible API proxy.

## Overview

This module provides an early version of a production-ready inference server that:

- Loads MLX-optimized models (text-only or vision-language models)
- Exposes an OpenAI-compatible API at `http://127.0.0.1:{MLX_PORT}/v1`
- Routes requests through LiteLLM proxy at `http://127.0.0.1:{PROXY_PORT}` for standardization
- Implements sophisticated prompt caching with block-based hash indexing
- Supports reasoning/thinking models with automatic output cleaning
- Provides session management with branching support for multi-agent workflows

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client                                  │
│                         (API Call)                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LiteLLM Proxy (Port 4000)                    │
│  - Standardizes OpenAI API params                               │
│  - Maps reasoning_effort → enable_thinking                      │
│  - Drops unsupported parameters                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MLX Server (Port 8080)                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    APIHandler                             │  │
│  │  - do_POST: /v1/chat/completions                          │  │
│  │  - do_GET: /v1/models                                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                Core Components                            │  │
│  │  • LRUPromptCache: Block-hashed prompt cache              │  │
│  │  • SessionIndex: Session/branch tracking                  │  │
│  │  • HealingStore: Message recovery for multi-agent         │  │
│  │  • ModelLock: Thread-safe GPU access                      │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MLX Model                                  │
│  - Text-only (mlx-lm) or VLM (mlx-vlm)                          │
│  - Quantized KV cache support (4-bit)                           │
└─────────────────────────────────────────────────────────────────┘
```

## Core Logic Flow

### 1. Request Handling

```python
POST /v1/chat/completions
  │
  ├─► Parse body & extract session context
  ├─► Determine enable_thinking from various sources
  ├─► Extract images (for VLM)
  │
  ├─► Apply Stateless Healing:
  │   └─► Check if assistant message matches cached hash
  │       └─► Restore full text with <think> if found
  │
  ├─► Prepare messages (normalize tool_calls, flatten content)
  │
  ├─► Tokenize prompt
  │
  ├─► Cache Lookup:
  │   ├─► Hash prompt into blocks (16-token chunks)
  │   ├─► Block chain lookup for prefix match
  │   └─► Return cache match type (exact/shorter/miss)
  │
  ├─► Generate response:
  │   ├─► Stream tokens with sampler
  │   ├─► Extract tool_calls from output
  │   ├─► Strip <think> reasoning blocks
  │   └─► Store result in cache
  │
  └─► Return response (stream/non-stream)
```

### 2. Prompt Cache System

The cache uses a **block-based hash chain** for efficient prefix matching:

- **Block Size**: 16 tokens (configurable via `PROMPT_CACHE_BLOCK_SIZE`)
- **Global Max Entries**: 24 (default, `PROMPT_CACHE_MAX_ENTRIES_GLOBAL`)
- **Per-Session Max**: 2 (default, `PROMPT_CACHE_MAX_ENTRIES_PER_SESSION`)
- **TTL**: 30 minutes (default, `PROMPT_CACHE_TTL_SECONDS`)

#### Cache Entry Structure

```python
@dataclass
class CacheEntry:
    prompt_cache: List[Any]  # MLX prompt cache object
    tokens: Tuple[int, ...]  # Full token sequence
    count: int               # Access frequency
    touched_at: float        # Last access timestamp
```

#### Block Hash Index

```python
_block_index: Dict[Tuple[str, bytes], set]
# Key: (model_name, chain_hash)
# Value: Set of token sequences containing this block
```

#### Matching Algorithm

1. Hash incoming prompt into blocks
2. Walk blocks backwards, find first matching block
3. Extract candidate cache entry
4. Extend match inside current block
5. Trim cache to match length

### 3. Session Management

Sessions track conversation context with lineage support for branching:

```python
@dataclass(frozen=True)
class SessionContext:
    session_id: str          # Unique session identifier
    parent_session_id: str   # Parent session (for branching)
    branch_id: str           # Branch identifier
    source: str              # Origin (request/derived)
```

#### SessionIndex Features

- **Lineage Chain**: Tracks parent→child relationships (max depth 8)
- **Anchor Points**: Strategic snapshots every 2048 tokens
- **LRU Pruning**: Idle sessions removed after `PROMPT_CACHE_SESSION_MAX_IDLE_SECONDS`

### 4. Healing Store

A stateless recovery mechanism for assistant messages in multi-agent sessions:

```python
HEALING_STORE: OrderedDict = OrderedDict()  # Max 2000 entries

# Hash-based lookup:
# Key: SHA-256(content + tool_calls)
# Value: Full text including <think> blocks
```

When an assistant message arrives:
1. Compute hash of stripped content
2. Check `HEALING_STORE` for match
3. If found, restore full text and remove `tool_calls` from message
4. Prevents Jinja double-rendering of tool calls

### 5. Metal Crash Fix

Prevents Apple Silicon GPU conflicts during VLM image processing:

```python
torch.backends.mps.is_built = lambda: False  # Force PyTorch to CPU
```

This "CPU quarantine" prevents Metal Command Buffer collisions between MLX and PyTorch during massive image prefills.

## Core Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| **MODEL_PATH** | `lmstudio-community/GLM-4.7-Flash-MLX-4bit` | HuggingFace repo or local path to MLX model |
| **MODEL_FAMILY** | `auto-detected` | Model family: `qwen3`, `glm4`, or `generic` |
| **MLX_HOST** | `127.0.0.1` | MLX server host |
| **MLX_PORT** | `8080` | MLX server port |
| **PROXY_PORT** | `4000` | LiteLLM proxy port |
| **PROXY_STARTUP_WAIT_SECONDS** | `2.0` | Wait time after proxy startup |
| **PROXY_MODEL_ID** | `openai/{model_path}` | Model ID exposed by proxy |

### Cache Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| **PROMPT_CACHE_MAX_ENTRIES_GLOBAL** | `24` | Max cache entries across all sessions |
| **PROMPT_CACHE_MAX_ENTRIES_PER_SESSION** | `2` | Max entries per session |
| **PROMPT_CACHE_TTL_SECONDS** | `1800` | Cache entry TTL (30 min) |
| **PROMPT_CACHE_SESSION_MAX_IDLE_SECONDS** | `1800` | Session idle timeout |
| **PROMPT_CACHE_BLOCK_SIZE** | `16` | Block size for hash indexing |
| **CACHE_USE_BLOCK_INDEX** | `true` | Enable block index lookup |
| **CACHE_CANONICALIZE_TOOL_CONTEXT** | `true` | Normalize tool context for caching |
| **CACHE_SESSION_PARTITIONING** | `true` | Partition cache by session |

### KV Cache Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| **MAX_KV_SIZE** | `262144` | Maximum KV cache size |
| **KV_GROUP_SIZE** | `64` | KV cache group size for GQA |
| **KV_BITS** | `4` | KV quantization bits (set `OFF` if crashes) |

### Sampling Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| **DEFAULT_TEMPERATURE** | `0.6` | Sampling temperature |
| **DEFAULT_TOP_P** | `0.95` | Top-p nucleus sampling |
| **DEFAULT_TOP_K** | `20` | Top-k sampling |
| **DEFAULT_MIN_P** | `0.00` | Minimum probability threshold |
| **DEFAULT_REPETITION_PENALTY** | `1.00` | Repetition penalty |
| **DEFAULT_REPETITION_CONTEXT_SIZE** | `256` | Context size for repetition penalty |
| **DEFAULT_MAX_TOKENS** | `16384` | Maximum output tokens |
| **DEFAULT_ENABLE_THINKING** | `true` | Enable reasoning/thinking output |

### Debug & Logging

| Variable | Default | Description |
|----------|---------|-------------|
| **ENABLE_REQUEST_LOGGING** | `true` | Enable request/response logging |
| **VLM_CACHE_DEBUG** | `true` | Enable VLM cache debugging |
| **LOG_ROOT** | `logs/` | Log file root directory |

## Core API Endpoints

### `GET /v1/models`

Returns available models:

```json
{
  "object": "list",
  "data": [{
    "id": "openai/local",
    "object": "model",
    "created": 1234567890,
    "owned_by": "mlx"
  }]
}
```

### `POST /v1/chat/completions`

Standard OpenAI chat completion with extensions:

**Request Body:**
```json
{
  "model": "openai/{model_path}",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 2048,
  "enable_thinking": true,
  "tools": [...]  // Optional function calling
}
```

**Response (Stream):**
```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"}}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_...","function":{"name":"write","arguments":"{..."}}]}]}

data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"finish_reason":"stop"}]}

data: [DONE]
```

### Reasoning/Thinking Support

The module automatically detects and handles reasoning output from models like Qwen3, GLM-4, and OpenClaw:

- **Input**: Accepts `enable_thinking`, `thinking`, `reasoning_effort`, `reasoning` fields
- **Output**: Automatically strips `https://...</think>` blocks from final response
- **Healing**: Stores full reasoning content for next-turn restoration

## VLM (Vision-Language Model) Support

When `mlx-vlm` is installed, the module automatically loads VLM models:

### Supported Model Types

```python
VLM_MODEL_TYPES = {
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe",
    "qwen3_5", "qwen3_5_moe", "qwen3_omni_moe",
    "glm4v", "glm4v_moe", "glm_ocr",
    "llava", "llava_next", "llava_qwen2", "llava_bunny",
    "idefics2", "idefics3", "mistral3",
    "gemma3", "gemma3n", "pixtral",
    "deepseek_vl_v2", "deepseekocr", "deepseekocr_2",
    "aya_vision", "cohere2_vision", "internvl_chat",
    "kimi_vl", "molmo", "molmo2", "smolvlm",
    "jina_vlm", "jvlm", "phi3_v", "paligemma",
    "florence2", "multi_modality", "mllama", "llama4",
}
```

### VLM Request Format

Images in messages:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."}}
      ]
    }
  ]
}
```

## Thread Safety & Concurrency

- **model_lock**: Global lock prevents concurrent GPU access (prevents Metal crashes)
- **prompt_cache_lock**: Protects cache modifications
- **console_lock**: Synchronizes terminal output
- **HEALING_STORE_LOCK**: Thread-safe healing store access

## Error Handling

### Metal Command Buffer Crash

If you encounter "Metal Command Buffer" errors:

1. The "CPU quarantine" fix (`torch.backends.mps.is_built = lambda: False`) is already applied
2. For VLM models, try disabling `mlx-vlm` or using `pip install mlx-vlm` without `[torch]` extra
3. Set `KV_BITS=OFF` if KV quantization causes crashes

### Model Loading Errors

- Text models: `from mlx_lm import load`
- VLM models: `from mlx_vlm import load` (requires `pip install mlx-vlm`)

## File Structure

```
mlx/
├── start-llm.py          # Main server implementation (~2400 lines)
├── .env                  # Environment configuration
├── .env.example          # Template with all configurable options
├── logs/                 # Request/response logs by date
│   └── 2026-02-28/
│       └── cache-session-<hash>.log
└── cache-debug/          # Cache debugging artifacts (if enabled)
```

## Performance Optimizations

1. **Block-based Hash Indexing**: O(1) block lookup vs O(n) sequential scan
2. **LRU + TTL**: Automatic cleanup of stale entries
3. **KV Cache Quantization**: 4-bit quantization reduces VRAM usage
4. **Session Partitioning**: Prevents cross-session cache pollution
5. **Canonicalization**: Normalizes volatile metadata for stable cache keys

## Troubleshooting

### Cache Not Working

- Check `PROMPT_CACHE_MAX_ENTRIES_GLOBAL` is set > 0
- Verify `CACHE_CANONICALIZE_TOOL_CONTEXT=true` for tool-heavy sessions
- Check logs for cache match type (exact/shorter/miss)

### VLM Loading Fails

```bash
# Install VLM support
pip install mlx-vlm

# For faster image processing (requires torchvision)
pip install mlx-vlm[torch]
```

### High Memory Usage

- Reduce `PROMPT_CACHE_MAX_ENTRIES_GLOBAL`
- Set `KV_BITS=OFF` to disable KV quantization
- Reduce `MAX_KV_SIZE` if appropriate

## Security Considerations

- **Local-Only**: Server binds to `127.0.0.1` by default
- **API Key**: LiteLLM proxy uses `local` API key (not for production)
- **No Authentication**: This is a local development/production inference server

## Future Enhancements

- [ ] Distributed cache sharing across multiple MLX servers
- [ ] Persistent cache storage (disk-based fallback)
- [ ] Model swapping without restart
- [ ] Streaming tool call execution feedback
- [ ] Batched request processing

