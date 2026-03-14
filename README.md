# MLX LLM Server

A high-performance, local LLM inference server optimised for Apple Silicon via the MLX framework. Exposes an OpenAI-compatible API through a LiteLLM proxy.

## Overview

- Loads MLX-optimised models (text-only or vision-language)
- Exposes an OpenAI-compatible API at `http://127.0.0.1:{MLX_PORT}/v1`
- Routes requests through a LiteLLM proxy at `http://127.0.0.1:{PROXY_PORT}` for parameter normalisation
- Two-layer prompt caching: global block-hash index + message-aware stable-prefix fallback
- Reasoning/thinking model support with automatic output cleaning and turn-over healing
- Multi-agent and sub-agent safe by design

## Request Path

```
Client
  │
  ▼
LiteLLM Proxy  (port 4000)
  • Normalises OpenAI params
  • Maps reasoning_effort → enable_thinking
  • Drops unsupported fields (drop_params=true)
  │
  ▼
MLX Server  (port 8080)
  • Healing → message prep → tokenise
  • Two-layer cache lookup (see below)
  • stream_generate / stream_generate_vlm
  • Cache insert + turn record update
  │
  ▼
MLX Model  (mlx-lm or mlx-vlm)
```

## Cache Architecture

The cache has two independent layers. The global layer always runs; the stable-prefix layer is a secondary fallback that only fires when the global layer falls short.

### Layer 1 — Global block-hash cache (`LRUPromptCache`)

Purely token-sequence-based. Session ID, agent identity, and message structure are irrelevant.

- Prompt tokens are split into 16-token blocks.
- Each block's hash is chained with the prior block's hash (SHA-256).
- The block index maps `(model, chain_hash) → set of cached token sequences`.
- On lookup: walk blocks backward from the longest matching block, extend within the block, trim cached KV to the matched length.
- On insert: `_cull_redundant_prefixes` removes shorter strict-prefix entries superseded by the new one (retains the longest).
- Eviction: cost-aware LRU — score = `age / (sqrt(length) × log(frequency+1))`. Short, old, rarely-used entries evicted first.

This layer correctly handles all multi-agent topologies, tool-result injections, mid-history insertions from CRON jobs, parallel sub-agent requests, and out-of-order arrivals — because it makes no assumptions about conversation structure.

### Layer 2 — Message-aware stable-prefix (`SESSION_TURN_STORE`)

An additive secondary layer that recovers cache hits the block-hash layer misses due to mid-history insertions or early whitespace drift.

**How it works:**

1. After each completed turn, `_update_session_turn_store` records `(messages, per_msg_token_lens)` for the session.
2. On the next request, `_message_diff(prev_msgs, curr_msgs)` finds the longest **contiguous leading prefix** of messages that are normalisation-identical between turns.
3. `_compute_msg_token_boundaries` converts that message count to an exact token boundary via cumulative `apply_chat_template` rendering.
4. If that token boundary exceeds what the global cache matched, a secondary `PROMPT_CACHE.fetch_nearest_cache` is issued against the stable prefix tokens. If it returns a better hit, the result replaces the global one.

**Multi-agent / sub-agent safety:**

`_update_session_turn_store` only writes if the incoming message list is a **strict append** of the existing record (all prior messages unchanged, new messages only at the tail). If any prior message differs — because the request is from a sub-agent, a parallel branch, or an out-of-order CRON injection — the write is silently skipped. The stored record always reflects the most recent linear orchestrator continuation. The global cache layer is completely unaffected.

**This layer never makes generation worse.** It only fires when `stable_prefix_token_len > global_matched_prefix_len`. If the secondary lookup also misses (e.g. the T−1 cache was evicted), the request falls through to the global result unchanged.

### Cache lookup flow

```
prompt_tokens
    │
    ├─► SESSION_TURN_STORE.get(session_id)
    │       └─► _message_diff(prev, curr)
    │               └─► stable_prefix_token_len  ─────────────────────┐
    │                                                                   │
    ├─► PROMPT_CACHE.fetch_nearest_cache(model, prompt_tokens)         │
    │       └─► matched_prefix_len  ──────────────────────────────┐    │
    │                                                              │    │
    │           if stable_prefix_token_len > matched_prefix_len ──┴────┘
    │               └─► PROMPT_CACHE.fetch_nearest_cache(
    │                       model, prompt_tokens[:stable_prefix_token_len])
    │                           └─► if better: use this result
    │
    └─► (prompt_cache, rest_tokens, match_type, matched_prefix_len)
```

### Eviction policy

`_cull_redundant_prefixes` runs on every insert. When a new N-token entry is stored, all existing entries whose token sequences are strict prefixes of the new one are identified. The longest of those prefix entries is spared (it can still serve requests shorter than N). The rest are deleted. This keeps the cache from filling with superseded short entries.

The cost-aware eviction (`_evict_optimal`) runs when the cache exceeds `PROMPT_CACHE_MAX_ENTRIES_GLOBAL`. Score = `age_seconds / (sqrt(len_tokens) × log1p(hit_count) + 1)`. Long frequently-used entries are strongly protected.

## Core Components

| Symbol | Type | Role |
|---|---|---|
| `LRUPromptCache` | class | Global block-hash KV cache store |
| `SessionIndex` | class | Per-session anchor tracking (lineage, branch keys) |
| `SESSION_TURN_STORE` | dict | Per-session last-turn message record for stable-prefix diff |
| `HEALING_STORE` | OrderedDict | SHA-256-keyed recovery of stripped `<think>` content |
| `Settings` | frozen dataclass | All configuration, loaded once from `.env` |
| `_message_diff` | function | Message-boundary diff returning stable leading prefix count |
| `_compute_msg_token_boundaries` | function | Exact per-message token boundaries via cumulative chat-template rendering |
| `_normalize_message_content_for_diff` | function | Per-message content normalisation for diff key only (whitespace strip; M4 rules TBD) |
| `_normalize_prompt_for_cache` | function | Rendered-prompt normalisation (scrubs timestamps, `<system-reminder>`, billing headers) |
| `_heal_messages` | function | Restores full `<think>` content to assistant messages before tokenisation |

## Environment Variables

### Core

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `lmstudio-community/GLM-4.7-Flash-MLX-4bit` | HuggingFace repo or local path |
| `MODEL_FAMILY` | auto-detected | `qwen3`, `glm4`, or `generic` |
| `MLX_HOST` | `127.0.0.1` | MLX server bind address |
| `MLX_PORT` | `8080` | MLX server port |
| `PROXY_PORT` | `4000` | LiteLLM proxy port |
| `PROXY_MODEL_ID` | `openai/{model_path}` | Model ID exposed by proxy |

### Cache

| Variable | Default | Description |
|---|---|---|
| `PROMPT_CACHE_MAX_ENTRIES_GLOBAL` | `24` | Max entries across all sessions |
| `PROMPT_CACHE_TTL_SECONDS` | `1800` | Entry TTL (30 min) |
| `PROMPT_CACHE_SESSION_MAX_IDLE_SECONDS` | `1800` | Session idle timeout |
| `PROMPT_CACHE_BLOCK_SIZE` | `16` | Block size for hash indexing |
| `CACHE_CANONICALIZE_TOOL_CONTEXT` | `true` | Scrub volatile metadata from rendered prompt |

### KV Cache

| Variable | Default | Description |
|---|---|---|
| `MAX_KV_SIZE` | `196608` | Maximum KV cache size in tokens |
| `KV_GROUP_SIZE` | `64` | Group size for GQA |
| `KV_BITS` | `off` | KV quantisation bits (`4`, `8`, or `OFF`) |

### Sampling

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_TEMPERATURE` | `0.1` | Sampling temperature |
| `DEFAULT_TOP_P` | `0.9` | Top-p nucleus sampling |
| `DEFAULT_TOP_K` | `40` | Top-k sampling |
| `DEFAULT_MIN_P` | `0.05` | Minimum probability threshold |
| `DEFAULT_REPETITION_PENALTY` | `1.08` | Repetition penalty |
| `DEFAULT_MAX_TOKENS` | `2048` | Maximum output tokens |
| `DEFAULT_THINKING` | `true` | Enable reasoning/thinking output |

### Logging

| Variable | Default | Description |
|---|---|---|
| `ENABLE_REQUEST_LOGGING` | `true` | Write per-request logs to `logs/` |
| `LOG_ROOT` | `logs/` | Log root directory |
| `VLM_CACHE_DEBUG` | `false` | Log token-level cache divergence for VLM |

## API

### `GET /v1/models`

Returns the loaded model as an OpenAI-compatible model list.

### `POST /v1/chat/completions`

Standard OpenAI chat completion. Additional fields accepted:

| Field | Type | Description |
|---|---|---|
| `enable_thinking` | bool | Override thinking on/off for this request |
| `reasoning_effort` | str | `"low"` / `"medium"` / `"high"` / `"off"` — mapped to `enable_thinking` |
| `session_id` | str | Stable identifier for the conversation (enables stable-prefix cache) |
| `parent_session_id` | str | Parent session for branch tracking |

Images are supported via `content` arrays with `type: "image_url"` or `type: "input_image"` (VLM models only).

## Healing Store

The healing store recovers `<think>` reasoning content that was stripped before returning to the client, so that the next turn's assistant message arrives at the model with its full original text (reasoning + answer), not the stripped version. This is critical for models that use chain-of-thought internally — feeding back a stripped assistant message breaks the KV cache match.

Flow:
1. After generation, if `<think>` was stripped, store `SHA-256(stripped_text) → full_raw_text` in `HEALING_STORE`.
2. On the next request, `_heal_messages` checks each incoming assistant message against the store.
3. If found, the full text (with `<think>`) is substituted before tokenisation.

`HEALING_STORE` is a bounded `OrderedDict` (max 2000 entries, LRU eviction). Access is protected by `HEALING_STORE_LOCK`.

## Threading Model

`ThreadingHTTPServer` spawns one thread per request. All requests queue on `model_lock` before touching the GPU — generation is strictly serialised. Cache reads/writes use `prompt_cache_lock`. Console output uses `console_lock`.

| Lock | Protects |
|---|---|
| `model_lock` | GPU / model access; serialises all generation |
| `prompt_cache_lock` | `PROMPT_CACHE`, `SESSION_TURN_STORE`, `SESSION_INDEX` |
| `console_lock` | Terminal output |
| `HEALING_STORE_LOCK` | `HEALING_STORE` |

## File Structure

```
mlx/
├── start-llm.py           # Entire server (~3100 lines)
├── install_and_run.py     # Bootstrapper: creates .venv, installs deps, execs server
├── requirements.txt       # pip dependencies
├── .env                   # Runtime configuration (copy from .env.example)
├── .env.example           # Template with all variables documented
├── scripts/
│   ├── probe_session.py   # Probe harness: 4 synthetic scenarios (normal/drift/semantic/insert)
│   ├── diff_turns.py      # Message-level differ for consecutive log entries
│   └── inspect_mlx_cache.py  # MLX KVCache API explorer (no model load required)
├── logs/
│   └── YYYY-MM-DD/
│       └── cache-session-<hash>.log   # Per-session request/response transcript
├── PLAN.md                # Active work plan, phase checklist, known limitations
├── DDR.md                 # Design decision record for SOTA cache architecture
└── AGENTS.md              # Instructions for agentic coding agents
```

## Troubleshooting

**Cache miss rate stays high after tool calls**
Check `stable_prefix_msg_count` in the logs. If it's 0 when you expect it to be >0, the system message may have drifted (FP-1) — check `_normalize_message_content_for_diff` and consider adding a normalisation rule for the specific volatile field.

**Metal / GPU crash**
`torch.backends.mps.is_built = lambda: False` is set at import time. If crashes recur with VLM models, try `pip install mlx-vlm` without the `[torch]` extra, or set `KV_BITS=OFF`.

**VRAM exhaustion**
Lower `MAX_KV_SIZE` or set `KV_BITS=4`.

**Server not responding**
Check both ports: 8080 (MLX engine) and 4000 (LiteLLM proxy). The proxy logs to stderr; check `/tmp/` for litellm process output.

**Wrong Python version**
The server requires Python 3.12. The bootstrapper (`install_and_run.py`) enforces this. Confirm with `python3.12 --version`.
