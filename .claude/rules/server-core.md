---
paths:
  - "start-llm.py"
---

# Server Core Rules

## Architecture: Dual Pipeline

The server uses two parallel pipelines. The model always receives original messages intact. The cache key uses canonical messages. They diverge at `_canonicalize_messages()` and never recombine.

```
Incoming Request -> _canonicalize_messages() -> (original, canonical)
  original -> apply_chat_template() -> model_tokens -> prefill+generate
  canonical -> apply_chat_template() -> _scrub_cache_key() -> prompt_tokens -> cache lookup
```

## Cache Layers

**Layer 1: Global Block-Hash Cache (`LRUPromptCache`)**
- 16-token blocks, SHA-256 chained hashes
- Longest-candidate selection (always pick deepest match)
- Cost-aware LRU eviction: `age / (sqrt(length) * log(frequency))`

**Layer 2: Message-Aware Stable-Prefix (`SESSION_TURN_STORE`)**
- Structural diff finds longest identical leading message prefix between turns
- Converts stable message count to exact token count via cumulative rendering
- Secondary lookup only fires when global match is weaker than stable prefix estimate
- Write guard: only records strict appends of prior message list (prevents sub-agent clobbering)

## Key Functions

| Function | Purpose |
|---|---|
| `_canonicalize_messages()` | Splits original/canonical at struct level. Replaces message_id, Inbound Context block, sub-agent stats |
| `_scrub_cache_key()` | Atomic line-scoped post-render scrubs (cch=, billing header, timestamps, system-reminder) |
| `_canonicalize_inbound_context_block()` | Produces identical output whether Inbound Context block is present or absent |
| `_kv_cache_offset(cache)` | Reads actual KV depth from `cache[0].offset`. MUST be used instead of canonical token counts for slicing model_tokens |
| `_assert_cache_key_safety()` | Invariant check: normalisation never deletes >10% of prompt |
| `_message_diff()` | Returns stable_prefix_count and change descriptors |
| `_stable_prefix_token_len()` | Converts stable message count to exact token boundary |
| `_compute_msg_token_boundaries()` | Cumulative `apply_chat_template(messages[:i+1])` for exact per-message token lengths |

## Thread Safety

- `model_lock`: Serialises GPU access (prefill + generate)
- `prompt_cache_lock`: Protects `PROMPT_CACHE`, `SESSION_TURN_STORE`, `SESSION_INDEX`
- `console_lock`: Terminal output serialisation

## Code Conventions

- All env vars read via `_env_str()` / `_env_int()` / `_env_bool()` helpers
- Settings frozen dataclass (`Settings`) built once at startup
- VLM support is optional (guarded by `mlx_vlm_available`)
- Tool call extraction: `TOOL_CALL_PATTERN`, `QWEN_FUNCTION_PATTERN` (XML/function tags)
- Metal crash fix: PyTorch MPS disabled at import time (`torch.backends.mps.is_built = lambda: False`)
