# Local LLM Serving Research — RTX 3090, Qwen3.5 27B, 200k Context

**Date:** 2026-04-01  
**Context:** Ubuntu, RTX 3090 (24GB VRAM), Qwen3.5 27B quantized, target 200k context.  
Currently achieving 132k at Q8 KV cache. Goal: 200k without quality loss.

---

## 1. TurboQuant — What It Actually Is

**Real paper.** Google Research, arXiv:2504.19874, accepted ICLR 2026.  
Authors include Vahab Mirrokni (Google Fellow).  
**KV cache quantization only** — not model weights.

### How it works

Two-stage algorithm:
1. Random rotation to induce Beta-distributed vectors → Lloyd-Max optimal scalar quantization per coordinate
2. 1-bit QJL (Quantized Johnson-Lindenstrauss) error correction to eliminate residual bias

Claims: quality-neutral at **3.5 bits**, marginal degradation at **2.5 bits**, up to 8x attention compute speedup on H100.

### Available precision levels (llama.cpp forks)

| Name    | Bits   | Compression vs FP16 |
|---------|--------|----------------------|
| turbo3  | ~3.25b | ~4.9x                |
| turbo4  | ~4.25b | ~3.8x                |

### Implementation status

**NOT merged into upstream llama.cpp** (as of 2026-04-01).

| Framework | Status |
|-----------|--------|
| llama.cpp upstream | Not merged. Feature request: ggml-org/llama.cpp#20977 |
| Community CUDA fork | `spiritbuun/llama-cpp-turboquant-cuda` — claims RTX 3090 (sm86) support |
| LM Studio | Not supported. Feature request: lmstudio-ai/lmstudio-bug-tracker#1719 |
| vLLM | No support |
| Ollama | No support |

Usage (in fork):
```bash
--cache-type-k turbo3 --cache-type-v turbo3 --flash-attn
```

To use today: build llama.cpp from the `spiritbuun` fork manually. Functional but unmerged.

Sources:
- https://arxiv.org/abs/2504.19874
- https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- https://github.com/ggml-org/llama.cpp/discussions/20969
- https://github.com/ggml-org/llama.cpp/issues/20977
- https://github.com/spiritbuun/llama-cpp-turboquant-cuda

---

## 2. NVIDIA's "2x Efficient Q4" — NVFP4

**What it is:** NVFP4 — 4-bit floating point (E2M1: 1 sign, 2 exponent, 1 mantissa). Two-level micro-block scaling (per-16-values FP8 scale + global FP32). **Model weight quantization**, not KV cache.

NVIDIA claims: 3.5x memory reduction vs FP16, ~3–4x throughput gain, <1% accuracy loss.

### Critical limitation: Blackwell GPUs only

Requires 5th-generation Tensor Cores (RTX 50xx, B100/B200).

**RTX 3090 (Ampere, compute capability 8.6) cannot use NVFP4.**  
Same applies to MXFP4 (requires CC ≥ 9.0). Confirmed: vllm-project/vllm#22422.

**Conclusion for RTX 3090:** NVFP4 is irrelevant until a hardware upgrade to Blackwell.

Sources:
- https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- https://build.nvidia.com/spark/nvfp4-quantization
- https://github.com/vllm-project/vllm/issues/22422

---

## 3. LM Studio Backend

- LM Studio uses **llama.cpp** as its inference backend (confirmed by NVIDIA blog)
- On Mac it also supports MLX as an alternative backend
- Bundles its own llama.cpp build — version not publicly disclosed
- The app layer is **not open source**; llama.cpp underneath is Apache 2.0

### KV cache quantization in LM Studio

**LM Studio does not expose KV cache quantization in its UI.**  
Config keys (`llamaKCacheQuantizationType`, `llamaVCacheQuantizationType`) exist in the llama.cpp layer but are not honored.  
See: lmstudio-ai/lmstudio-bug-tracker#186 (open since 2024).

**Implication:** Do not rely on LM Studio for KV cache quant control. Use llama.cpp directly.

Sources:
- https://lmstudio.ai/docs/app
- https://blogs.nvidia.com/blog/rtx-ai-garage-lmstudio-llamacpp-blackwell/
- https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/186

---

## 4. Framework Comparison for RTX 3090 + Single-User + 200k Context

### KV Cache Quantization Support

| Framework    | KV Cache Quant Options            | Notes |
|--------------|-----------------------------------|-------|
| llama.cpp    | f16, q8_0, q4_0, q4_1, q5_0, q5_1, IQ4_NL + TurboQuant (fork) | Most flexible. `--cache-type-k` / `--cache-type-v` flags |
| Ollama       | f16, q8_0 only                    | `OLLAMA_KV_CACHE_TYPE` env var |
| vLLM         | fp8_e4m3, fp8_e5m2                | **Requires compute capability > 8.9 — RTX 3090 is 8.6. Not available.** |
| LM Studio    | None exposed in UI                | Underlying llama.cpp supports it, UI does not |

### vLLM is not the right choice here

vLLM is designed for multi-user, high-concurrency serving. For single-user local inference, llama.cpp delivers better throughput. Additionally, vLLM's FP8 KV cache is unsupported on RTX 3090.

**Winner for this use case: llama.cpp direct** (`llama-server` or `llama-cli`).

Sources:
- https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/
- https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case
- https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/

---

## 5. VRAM Math — Why 132k Is the Current Limit

Qwen3.5 27B architecture (approx: 64 layers, 8 KV heads, 128 head_dim):

```
KV VRAM = 2 × layers × tokens × kv_heads × head_dim × bytes_per_element
        = 2 × 64 × tokens × 8 × 128 × bytes
```

| Context | Q8 KV (1B) | Q4 KV (0.5B) | turbo3 (~0.4B) |
|---------|-----------|-------------|----------------|
| 132k    | ~2.0 GB   | ~1.0 GB     | ~0.8 GB        |
| 165k    | ~2.5 GB   | ~1.25 GB    | ~1.0 GB        |
| 200k    | ~3.0 GB   | ~1.5 GB     | ~1.2 GB        |

Model weights at Q4_K_M ≈ 14–15 GB for 27B → ~9–10 GB free for KV cache + overhead.

At Q8 KV + 200k context the KV cache alone is only ~3 GB. The 132k limit is likely due to workspace tensor overhead, not pure KV size. Running llama.cpp directly (instead of LM Studio) may already unlock 200k at Q8 by reducing overhead.

---

## 6. Combining Weight Quantization + TurboQuant KV Cache

**Yes — fully orthogonal.** Model weight quantization (GGUF format) and KV cache quantization are separate memory regions. You already combine them when running Q4_K_M GGUF with `--cache-type-k q8_0`.

TurboQuant just replaces the KV dtype:
```bash
llama-server -m model-q4km.gguf \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  --flash-attn --ctx-size 200000
```

NVFP4 + TurboQuant combination would also be valid in principle, but NVFP4 doesn't run on RTX 3090.

---

## Action Plan

| Step | Action |
|------|--------|
| 1 (now) | Switch from LM Studio to `llama-server` directly. Full KV cache flag control. |
| 2 (now) | Try `--cache-type-k q8_0 --cache-type-v q8_0 --ctx-size 200000 --flash-attn` — likely fits on 3090. |
| 3 (when ready) | Build `spiritbuun/llama-cpp-turboquant-cuda` fork for TurboQuant KV cache. |
| 4 (track) | Watch ggml-org/llama.cpp#20977 for upstream TurboQuant merge. |
| 5 (skip) | Ignore NVFP4/MXFP4 — Blackwell only, irrelevant until hardware upgrade. |
