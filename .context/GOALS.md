# Goals & Strategic Constraints

## North Star

Build the best possible KV-cache-aware inference server for Apple Silicon.

**"Best" means:**
- Maximum cache hit rate across the full range of real client behaviour -- pure tail-appends, mid-history tool-result injections, multi-agent sub-session spawns, context compaction, heartbeats, silent system-prompt drift.
- Minimum prefill latency. On Apple Silicon with a 27B model, cold 10k-token prefill costs ~42 seconds. Cache hit rate IS the product.
- SOTA engineering quality. Design informed by vLLM, SGLang, and production inference systems. Established patterns applied to the MLX/Apple Silicon constraint set.

## Guiding Principles

1. **Evidence before code.** No caching changes without log-confirmed reproduction.
2. **Systemic over surgical.** Architecture that handles the class of problems, not each instance.
3. **Think in structures, not strings.** Diff and cache at the message-struct level, not serialised strings.
4. **Reuse as much KV state as physically possible.** Even if middle changed, the prefix to that point is golden.
5. **Observability is first-class.** If we can't measure it precisely, we can't improve it.

## Constraints

| Rule | Detail |
|---|---|
| No model blinding | All semantic content always reaches the model intact |
| No gut-feel patches | Every code change must cite a specific log/diff |
| No cross-session leakage | Cache isolation between sessions maintained |
| Single-file architecture | Changes stay in `start-llm.py` unless explicitly agreed |
| Python 3.12 only | No 3.13+ syntax |
| Thread safety | All shared state protected by locks |
| Normalisation is pre-hash only | Model always sees real content |
| Agility | Must handle arbitrary clients without per-client patches |
