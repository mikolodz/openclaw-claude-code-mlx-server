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
9. **Read `start-llm.py` end-to-end before editing it.** Not a grep, not a scan — the whole file. Cache/normalisation logic is tightly coupled and remote edits have non-local effects. Hard requirement.

## IMPORTANT: Alignment Rule

When the owner states a hypothesis, **verify it first** before forming a counter-opinion. Check with actual evidence from logs, code, or data. Only after genuinely disproving the hypothesis with hard evidence may you present a different conclusion. Lead with evidence, not disagreement.

## IMPORTANT: Change Protocol

Every non-trivial change to cache, normalisation, or the request pipeline follows this protocol:

1. **Portion sizing.** Group related fixes into one coherent portion. Don't work one ticket at a time (too granular = churn); don't bundle unrelated fixes (blast radius too large). A portion is "all the items that share a root cause or a test surface."
2. **Read the whole file first.** See rule 9.
3. **Write a short spec** before touching code. The spec must answer three questions explicitly:
   - **Design quality.** Is this production-grade, or a prototype patch? Name the abstraction, the invariant, the failure mode it closes.
   - **Root cause vs. cover-up.** Does this *solve* the issue, or does it silence a symptom while the real bug remains? If it's a mitigation, say so and link the root-cause ticket.
   - **Regression surface.** What else in the design can this break? What new bugs could it introduce? Which existing invariants does it touch?
4. **Implement** only after the spec is agreed.
5. **Verify.** See Testing section below.

## Testing

There is no unit test suite. Verification is empirical via real agentic traffic.

**Preferred harness: real `pi` CLI** (our favourite agentic client for local models). `pi_like_probe.py` is a *synthetic* probe — it does NOT reproduce the pi client's echo pattern (it re-sends the server's stripped response as content + reasoning_content the way pi actually does), so probe numbers alone can mislead. Always validate cache work against real `pi` traffic.

### Running real pi against the local server

The `pi` CLI (installed as `/opt/homebrew/bin/pi`, `@mariozechner/pi-coding-agent`) is pre-configured via `~/.pi/agent/{settings.json,models.json}` to route to `http://127.0.0.1:4000/v1` via provider `omlx`. Sessions are JSONL files in `~/.pi/agent/sessions/` or `--session-dir <path>`.

Standard 3-turn smoke protocol (multi-turn cache verification):
```bash
mkdir -p /tmp/pi-smoke && rm -rf /tmp/pi-smoke/*
# Turn 1 (cold start)
pi -p --no-tools --session-dir /tmp/pi-smoke --thinking off "Reply with a single word: hello"
# Turn 2 (must hit cache)
pi -p --no-tools --session-dir /tmp/pi-smoke -c --thinking off "Reply with a single word: world"
# Turn 3 (must hit cache)
pi -p --no-tools --session-dir /tmp/pi-smoke -c --thinking off "Reply with a single word: foo"
```
Flags: `-p` non-interactive, `-c` continue the latest session in the dir, `--no-tools` keeps the prompt small, `--thinking off` applies to the CLIENT's tool planning (the server still emits reasoning_content deltas — this is a knob for pi's internal flow, NOT the MLX server).

Expected post-fix (Qwen3.6 VLM, preserve_thinking=True): T1 miss @ 0%, T2+ ≥ ~95% `shorter`-hit. Watch the server stdout for the `📨 … Cache: XX%` line and `logs/<date>/cache-session-*.log` for the per-request JSON. If T2 hits <50% or bumps `hybrid_trim_miss`, diff turn 1 vs turn 2 via `python scripts/diff_turns.py logs/<date>/cache-session-*.log 0 1` — the first place a regression shows is the assistant-message rendering (preserve_thinking + echoed `reasoning_content` is the known failure mode, see LESSONS.md).

### Other verification helpers
- `python scripts/probe_session.py` — 4-scenario cache validation (normal, drift, semantic, insert).
- `scripts/diff_turns.py` — diff session prompts turn-by-turn.
- `scripts/inspect_mlx_cache.py` — inspect in-memory cache state.
- `logs/<date>/cache-session-*.log` — per-session cache trace; first place to look when a fix may have regressed hit rate.

Quick liveness check:
```bash
curl http://127.0.0.1:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/local","messages":[{"role":"user","content":"ping"}],"stream":false}'
```

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
