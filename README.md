# MLX OpenAI-Compatible Local Server

Lightweight local bridge for running an `mlx-lm` model behind an OpenAI-compatible API and exposing it through a LiteLLM proxy.

## What this repo provides

- `start-llm.py` - MLX chat completion backend (`/v1/chat/completions`) + model listing (`/v1/models`)
- LiteLLM proxy bootstrap on a second port for OpenClaw / Claude-compatible clients
- `.env` driven configuration (no manual export needed to run the server)
- Prompt-cache-session logging (single log file per cache session, not per request)
- `install_and_run.py` one-click setup + run flow

## Requirements

- macOS (Apple Silicon for MLX)
- Python `3.12` (required)

## Install and run

From this directory:

```bash
python3.12 install_and_run.py
```

What it does:

1. Creates `.env` from `.env.example` if missing
2. Creates `.venv` with Python 3.12 if missing
3. Installs dependencies from `requirements.txt`
4. Starts `start-llm.py` with the venv interpreter

## Configuration (`.env`)

Edit `.env` to control:

- model selection (`MODEL_PATH`, optional `PROXY_MODEL_ID`)
- server endpoints (`MLX_HOST`, `MLX_PORT`, `PROXY_PORT`)
- prompt cache behavior (`PROMPT_CACHE_MAX_SIZE`, `PROMPT_CACHE_TTL_SECONDS`)
- generation defaults (`DEFAULT_*`, `DEFAULT_MAX_TOKENS`)
- logging (`ENABLE_REQUEST_LOGGING`, `LOG_ROOT`)
- KV cache options (`MAX_KV_SIZE`, `KV_GROUP_SIZE`, `KV_BITS`)

Important KV note:

- Not all MLX models support KV cache quantization.
- If generation errors out with KV/cache-related issues, set:

```env
KV_BITS=OFF
```

## OpenClaw setup

When using a custom model in OpenClaw onboard:

- choose custom model/provider endpoint
- set endpoint to your LiteLLM proxy, typically `http://127.0.0.1:4000`
- use the full model name, for example:

```text
openai/lmstudio-community/GLM-4.7-Flash-MLX-4bit
```

In JSON-based config this corresponds to an OpenAI-compatible provider with:

- `baseUrl`: `http://127.0.0.1:4000`
- `apiKey`: `local`

## Claude Code setup

Keep this server running, then in your shell set:

```bash
export ANTHROPIC_BASE_URL="http://localhost:4000"
export ANTHROPIC_API_KEY="local"
claude
```

## Files

- `.env.example` - template config with defaults and comments
- `.env` - local active config (ignored by git)
- `requirements.txt` - pip dependencies
- `install_and_run.py` - one-command bootstrap and run script
