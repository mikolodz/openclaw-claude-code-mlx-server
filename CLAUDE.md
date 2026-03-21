# MLX LLM Server

Working directory: `/Users/michalkolodziej/mlx`

## Project Identity

A KV-cache-aware LLM inference server for Apple Silicon via MLX. Exposes an OpenAI-compatible API through a LiteLLM proxy. The core value proposition is **maximum cache hit rate** to minimize prefill latency (~42s per 10k tokens cold on a 27B model).

## Key Files

| File | Purpose |
|---|---|
| `start-llm.py` | Entire server (~3500 LOC): API handler, prompt cache, sessions, dual pipeline |
| `install_and_run.py` | Bootstrap: creates `.venv`, installs deps, execs server |
| `.env` / `.env.example` | Runtime config (model path, cache sizes, ports) |
| `requirements.txt` | pip deps: mlx, mlx-lm, mlx-vlm, litellm[proxy], python-dotenv |
| `scripts/` | Analysis tools: `diff_turns.py`, `inspect_mlx_cache.py`, `probe_session.py` |
| `scripts/test_openclaw_integration.py` | Automated OpenClaw integration test harness |

## Build & Run

```bash
# One-command (creates .venv, installs deps, starts server)
python3.12 install_and_run.py

# Manual
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python start-llm.py

# Verify
curl -s http://127.0.0.1:4000/v1/models
```

**Request path:** Client -> LiteLLM Proxy (port 4000) -> MLX Server (port 8080) -> MLX Model

## IMPORTANT: Non-Negotiable Rules

1. **The model must NEVER be blinded.** Tool results, `<think>` blocks, file contents -- all semantic content reaches the model 100% intact. A cache hit that hides content is worse than a miss.
2. **Separate cache key from model input.** Canonical pipeline for cache lookup, original pipeline for model. The two NEVER recombine after `_canonicalize_messages()`.
3. **Normalisation operates on structured data, not rendered strings.** Volatile fields are normalised at message-struct level before `apply_chat_template`. Post-render scrubs are atomic, line-scoped only. NO `re.DOTALL` spanning section boundaries -- ever.
4. **No gut-feel patches.** Every change to caching/normalisation must cite a specific log-confirmed failure mode. Read `.context/LESSONS.md` before touching cache logic.
5. **Evidence before code.** No architecture changes without log-confirmed reproduction.
6. **Python 3.12 strict.** No other version supported.
7. **Single-file server.** All server logic stays in `start-llm.py` unless explicitly agreed.
8. **Thread safety.** All shared state (`PROMPT_CACHE`, `SESSION_TURN_STORE`, `HEALING_STORE`) protected by locks.

## IMPORTANT: Alignment Rule

When the owner states a hypothesis, **verify it first** before forming a counter-opinion. Check with actual evidence from logs, code, or data. Only after genuinely disproving the hypothesis with hard evidence may you present a different conclusion. Lead with evidence, not disagreement.

## Testing

There is no test suite. Verify via:
```bash
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/local","messages":[{"role":"user","content":"ping"}],"stream":false}'
```

For cache validation: `python scripts/probe_session.py` (4 scenarios: normal, drift, semantic, insert).

## Active Work

@.context/TASKS.md

## Deep Context

| File | When to read | Loading mechanism |
|---|---|---|
| `.context/TASKS.md` | Always (current/upcoming work) | `@` import above |
| `.context/GOALS.md` | Strategic decisions, scope questions | On demand |
| `.context/LESSONS.md` | Before touching cache logic | On demand |
| `.claude/history/` | When referencing completed phases (0-8) | On demand |
| `.claude/rules/server-core.md` | Auto-loads when editing `start-llm.py` | Path-scoped rule |
| `.claude/rules/scripts.md` | Auto-loads when editing `scripts/` | Path-scoped rule |
| `.claude/skills/test-*/` | When running test protocols | `/skill-name` invocation |
