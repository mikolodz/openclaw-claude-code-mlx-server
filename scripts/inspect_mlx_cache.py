#!/usr/bin/env python3
"""
MLX KV Cache API inspection script.
Tests what operations are physically possible on KVCache / QuantizedKVCache objects.
Loads a text-only prefix from the language model component of a VLM if needed.
"""

import sys
import mlx.core as mx


def main() -> None:
    model_path = "mlx-community/Qwen3.5-35B-A3B-4bit"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    print(f"Loading {model_path}...")

    # Qwen3.5 is a VLM — use mlx_vlm loader
    try:
        from mlx_vlm import load as load_vlm

        result = load_vlm(model_path)
        model = result[0]
        processor = result[1]
        # For text-only prefill, grab the language model sub-module
        lm = getattr(model, "language_model", model)
        tokenizer = getattr(processor, "tokenizer", processor)
        print("Loaded via mlx_vlm (VLM model)")
    except Exception as e:
        print(f"mlx_vlm load failed: {e}")
        print("Falling back to mlx_lm...")
        from mlx_lm import load as load_lm

        result_lm = load_lm(model_path)
        lm = result_lm[0]
        tokenizer = result_lm[1]
        print("Loaded via mlx_lm")

    # Build a short prompt and tokenize it
    prompt = "Hello world this is a test prompt for cache inspection"
    from mlx_lm.models.cache import make_prompt_cache

    tokens = tokenizer.encode(prompt)
    print(f"\nPrompt: {repr(prompt)}")
    print(f"Prompt tokens ({len(tokens)}): {tokens}")

    print("\n--- Forward pass to populate KV cache ---")
    cache = make_prompt_cache(lm)
    prompt_tokens_arr = mx.array(tokens)[None]  # shape [1, T]
    logits = lm(prompt_tokens_arr, cache=cache)
    mx.eval(logits)

    print(f"\nCache type: {type(cache).__name__}")
    print(f"Cache length (layers): {len(cache)}")

    if not cache:
        print("ERROR: cache is empty after forward pass")
        return

    c0 = cache[0]
    print(f"\nLayer 0 cache type: {type(c0).__name__}")
    print(f"  offset (tokens stored): {c0.offset}")

    if c0.keys is not None:
        print(f"  keys type:  {type(c0.keys)}")
        print(f"  keys shape: {c0.keys.shape}  (B, n_kv_heads, alloc_seq, head_dim)")
        print(f"  values shape: {c0.values.shape}")
        # The allocated buffer may be larger than offset (step=256 allocation)
        print(
            f"  allocated seq slots: {c0.keys.shape[2]}  (offset={c0.offset}, step=256)"
        )
    else:
        print("  keys: None (QuantizedKVCache — stored as tuples)")
        if hasattr(c0, "group_size"):
            print(f"  group_size={c0.group_size}, bits={c0.bits}")

    # --- Trimmability ---
    print("\n--- Trim support ---")
    print(f"  is_trimmable(): {c0.is_trimmable()}")
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

    print(f"  can_trim_prompt_cache(cache): {can_trim_prompt_cache(cache)}")

    # Test actual trim: remove last 3 tokens
    trim_n = 3
    print(f"\n  Trimming {trim_n} tokens from tail...")
    trimmed = trim_prompt_cache(cache, trim_n)
    print(f"  trim_prompt_cache returned: {trimmed}")
    print(
        f"  Layer 0 offset after trim: {cache[0].offset}  (expected {len(tokens) - trim_n})"
    )

    # --- Raw array manipulation (segment surgery feasibility) ---
    print("\n--- Segment surgery feasibility test ---")
    print(
        "Goal: can we copy specific token positions out of the raw keys/values arrays?"
    )

    # Re-run forward pass to get a fresh cache
    cache2 = make_prompt_cache(lm)
    logits2 = lm(prompt_tokens_arr, cache=cache2)
    mx.eval(logits2)
    c0 = cache2[0]

    try:
        if c0.keys is not None:
            # Standard KVCache: keys shape is [B, heads, alloc_seq, head_dim]
            # Actual filled region is [B, heads, :offset, head_dim]
            k_full = c0.keys[..., : c0.offset, :]  # shape [B, heads, T, head_dim]
            v_full = c0.values[..., : c0.offset, :]
            print(
                f"  Active K region shape: {k_full.shape}  (T={c0.offset} = {len(tokens)} tokens)"
            )

            # Slice tokens 2-5 out (keep prefix [0:2] and suffix [5:])
            T = c0.offset
            keep_indices = list(range(0, 2)) + list(range(5, T))
            k_splice = mx.concatenate([k_full[..., :2, :], k_full[..., 5:, :]], axis=2)
            v_splice = mx.concatenate([v_full[..., :2, :], v_full[..., 5:, :]], axis=2)
            mx.eval(k_splice, v_splice)
            print(f"  Spliced K shape: {k_splice.shape}  (removed tokens 2-4)")

            # Can we write spliced data back into a new cache via the .state setter?
            import copy

            c0_copy = copy.deepcopy(cache2[0])
            c0_copy.state = (k_splice, v_splice)  # uses the @state.setter
            print(
                f"  After state= setter: offset={c0_copy.offset}, keys.shape={c0_copy.keys.shape}"
            )
            print(
                "  -> Segment surgery via concatenation + state setter: CONFIRMED FEASIBLE"
            )
        else:
            print("  QuantizedKVCache detected — keys are tuples of quantized arrays.")
            print("  Surgery requires dequantize -> splice -> re-quantize.")
            k_dq = mx.dequantize(*c0.keys[:3])  # keys is (data, scales, biases)
            print(f"  Dequantized keys shape: {k_dq.shape}")
            print(
                "  -> Segment surgery via dequantize+splice+quantize: feasible but lossy"
            )

    except Exception as e:
        print(f"  Segment surgery test FAILED: {e}")
        import traceback

        traceback.print_exc()

    print("\n=== Summary ===")
    print("  trim_prompt_cache:    tail-trim only (subtracts from .offset)")
    print("  KVCache.state setter: allows arbitrary [B,heads,T,dim] tensor injection")
    print("  mx.concatenate:       can slice and reassemble key/value arrays")
    print(
        "  Conclusion: segment surgery (prefix extraction + injection) is PHYSICALLY POSSIBLE"
    )
    print("  Caveat: RoPE-based models require position-consistent reuse.")
    print(
        "  Caveat: QuantizedKVCache surgery requires dequantize/re-quantize round-trip."
    )


if __name__ == "__main__":
    main()
