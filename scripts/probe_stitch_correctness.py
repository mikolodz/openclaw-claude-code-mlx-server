#!/usr/bin/env python3
"""
G4 §7.1 sanity probe — per-layer snapshot + sliceable-layer-trim stitching.

Three phases, each with a strict bit-exact pass/fail:

  Phase 1 (§7.2-ish): Does copy.deepcopy(cache) produce an independent state?
    Control:     prefill X, generate N tokens.
    Experiment:  prefill X, deepcopy, generate N tokens from the deepcopy.
    Pass if tokens match bit-exact.

  Phase 2: Does per-layer-trim of a longer cache produce the same state as a
           fresh prefill to the shorter depth?
    Control:     prefill X, generate N tokens → tokens_ctrl.
    Experiment:  prefill X+Y (longer), per-layer-trim sliceable layers to |X|,
                 use snapshot_X for non-sliceable layers (both trimmed to |X|).
                 Then generate N tokens.
    Pass if tokens match bit-exact.

  Phase 3 (the real test): Full cross-source stitch.
    Control:     prefill X, generate N tokens from continuation V.
    Experiment:  take snapshot_X (prefill X only) for non-sliceable layers,
                 take cache_B (prefill X+Y_b) per-layer-trimmed to |X| for
                 sliceable layers, stitch, generate N from V.
    Pass if tokens match bit-exact.

If Phase 1 fails → deepcopy is broken (need mx.eval + manual copy).
If Phase 2 fails → per-layer trim is broken or ArraysCache state is
                   path-dependent.  G4v2 design collapses; fall back to v1.
If Phase 3 fails but 1+2 pass → cross-layer coupling we missed.

Usage:
  python3 scripts/probe_stitch_correctness.py [model_path]

Default model: reads .env MODEL_PATH.
"""

from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import mlx.core as mx

_SLICEABLE_CLASSES = frozenset({
    "KVCache", "BatchKVCache", "QuantizedKVCache",
    "TurboQuantKVCache", "BatchTurboQuantKVCache",
    "RotatingKVCache",  # from mlx_lm
})


def _is_sliceable_layer(c: Any) -> bool:
    """Return True iff the layer cache can be per-layer trimmed safely."""
    return type(c).__name__ in _SLICEABLE_CLASSES


def _classify_layers(cache: List[Any]) -> Tuple[List[int], List[int]]:
    sliceable = [i for i, c in enumerate(cache) if _is_sliceable_layer(c)]
    non_sliceable = [i for i, c in enumerate(cache) if not _is_sliceable_layer(c)]
    return sliceable, non_sliceable


def _per_layer_trim_sliceable(cache: List[Any], target_depth: int) -> None:
    """In-place per-layer trim for sliceable layers only.  Non-sliceable layers untouched.

    Works for KVCache (has .keys/.values/.offset).  For QuantizedKVCache we
    use the .state setter on (K, V) slices.
    """
    for c in cache:
        if not _is_sliceable_layer(c):
            continue
        if getattr(c, "keys", None) is not None:
            cur = int(getattr(c, "offset", 0) or 0)
            if cur <= target_depth:
                continue
            c.keys = c.keys[..., :target_depth, :]
            c.values = c.values[..., :target_depth, :]
            c.offset = target_depth
        else:
            # QuantizedKVCache — keys/values are tuples.
            raise NotImplementedError("QuantizedKVCache path not exercised in this probe")
    mx.eval([c.state for c in cache if hasattr(c, "state") and _is_sliceable_layer(c)])


def _deepcopy_cache(cache: List[Any]) -> List[Any]:
    """Deepcopy with eager materialization, to defeat MLX lazy-array aliasing."""
    # Force materialization of any pending lazy compute first.
    for c in cache:
        st = getattr(c, "state", None)
        if st is not None:
            try:
                mx.eval(st)
            except Exception:
                pass
    return copy.deepcopy(cache)


def _make_cache(lm: Any) -> List[Any]:
    from mlx_lm.models.cache import make_prompt_cache
    return make_prompt_cache(lm)


def _forward_logits(lm: Any, arr: mx.array, cache: List[Any]) -> mx.array:
    out = lm(arr, cache=cache)
    logits = out.logits if hasattr(out, "logits") else out
    mx.eval(logits)
    return logits


def _prefill(lm: Any, token_ids: List[int], cache: List[Any]) -> mx.array:
    arr = mx.array(token_ids, dtype=mx.int32)[None]
    return _forward_logits(lm, arr, cache)


def _greedy_generate(
    lm: Any, prompt_ids: List[int], cache: List[Any], n_new: int
) -> List[int]:
    """Greedy argmax for n_new tokens, starting from prompt_ids (prefilled into cache).

    prompt_ids is processed as a single forward pass if cache.offset == 0,
    else treated as continuation tokens (one forward of shape [1, len(prompt_ids)]).
    """
    arr = mx.array(prompt_ids, dtype=mx.int32)[None]
    logits = _forward_logits(lm, arr, cache)
    next_tok = int(mx.argmax(logits[0, -1, :]).item())
    out: List[int] = [next_tok]
    for _ in range(n_new - 1):
        a = mx.array([[next_tok]], dtype=mx.int32)
        logits = _forward_logits(lm, a, cache)
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        out.append(next_tok)
    return out


def _stitch(
    snapshot: List[Any],
    full_cache: List[Any],
    non_sliceable: List[int],
    sliceable: List[int],
    target_depth: int,
) -> List[Any]:
    """Build a stitched cache at target_depth from two sources.

    non_sliceable layers come from `snapshot` (which is pre-trimmed to target_depth).
    sliceable layers come from `full_cache` deep-copied and per-layer-trimmed.
    """
    assert len(snapshot) == len(full_cache), "layer count mismatch"
    nset = set(non_sliceable)
    out: List[Any] = [None] * len(snapshot)
    for i in nset:
        out[i] = copy.deepcopy(snapshot[i])
    # Sliceable: deepcopy from full_cache then trim in-place.
    trimmed_full = _deepcopy_cache(full_cache)
    _per_layer_trim_sliceable(trimmed_full, target_depth)
    for i in sliceable:
        out[i] = trimmed_full[i]
    return out


def _load_model(model_path: str):
    print(f"[load] {model_path}", flush=True)
    t0 = time.time()
    from mlx_vlm import load as load_vlm
    model, processor = load_vlm(model_path)
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    lm = getattr(model, "language_model", model)
    return lm, tokenizer


def _tokenize_prefix(tokenizer, n_target: int = 1024) -> List[int]:
    """Build a deterministic ~n_target-token prefix via chat template."""
    # Long, self-coherent system/user to approximate real traffic.
    sys_content = (
        "You are a meticulous debugging assistant specializing in cache correctness "
        "for large language model inference servers.  Always cite evidence, never "
        "guess, and be concise.  "
    ) * 8
    usr_content = (
        "Explain, in technical detail, the difference between paged attention and "
        "radix attention, how each handles prefix sharing, and why hybrid models "
        "with linear-attention layers complicate per-position trim operations. "
        "Cover: block granularity, hash-keyed blocks, reference counting, trie "
        "structure, node splitting on divergence, and why ArraysCache state is "
        "path-dependent when Mamba-like recurrences are involved.  "
    ) * 4
    try:
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": usr_content},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        ids = tokenizer.encode(rendered)
    except Exception:
        ids = tokenizer.encode(sys_content + "\n\n" + usr_content)
    # Truncate or pad to roughly n_target tokens.
    if len(ids) > n_target:
        ids = ids[:n_target]
    print(f"[prefix] {len(ids)} tokens", flush=True)
    return ids


def _cache_offsets(cache: List[Any]) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    for c in cache:
        off = getattr(c, "offset", None)
        out.append(int(off) if isinstance(off, (int, float)) else None)
    return out


# ------------------------- phases --------------------------------------


def phase1_deepcopy(lm, prefix_ids: List[int], n_new: int = 16) -> bool:
    """Prefill then decode-only; deepcopy occurs between prefill and decode.

    Control:  cache = make(); prefill(prefix); loop decode N.
    Experiment: cache_src = make(); prefill(prefix); cache_copy = deepcopy;
                loop decode N on cache_copy.  Must match control token-by-token.
    """
    print("\n=== Phase 1 — deepcopy invariance ===", flush=True)

    # Control.
    cache_ctrl = _make_cache(lm)
    logits = _prefill(lm, prefix_ids, cache_ctrl)
    next_tok = int(mx.argmax(logits[0, -1, :]).item())
    ctrl = [next_tok]
    for _ in range(n_new - 1):
        a = mx.array([[next_tok]], dtype=mx.int32)
        logits = _forward_logits(lm, a, cache_ctrl)
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        ctrl.append(next_tok)

    # Experiment.
    cache_src = _make_cache(lm)
    logits_src = _prefill(lm, prefix_ids, cache_src)
    cache_cp = _deepcopy_cache(cache_src)
    next_tok = int(mx.argmax(logits_src[0, -1, :]).item())
    exp = [next_tok]
    for _ in range(n_new - 1):
        a = mx.array([[next_tok]], dtype=mx.int32)
        logits = _forward_logits(lm, a, cache_cp)
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        exp.append(next_tok)

    ok = ctrl == exp
    print(f"  control  : {ctrl[:8]}{'...' if len(ctrl) > 8 else ''}", flush=True)
    print(f"  deepcopy : {exp[:8]}{'...' if len(exp) > 8 else ''}", flush=True)
    print(f"  result   : {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        for i, (a, b) in enumerate(zip(ctrl, exp)):
            if a != b:
                print(f"  first divergence at position {i}: ctrl={a} exp={b}", flush=True)
                break
    return ok


def phase2_trim(lm, prefix_ids: List[int], n_new: int = 16, filler_n: int = 32) -> bool:
    """Does per-layer-trim of a longer cache produce state ≡ fresh prefill to |prefix|?

    snapshot_short = prefill(prefix).
    cache_long     = prefill(prefix); then decode filler_n tokens (cache now
                                              at depth |prefix| + filler_n).
    stitch         = non-sliceable layers from snapshot_short,
                     sliceable layers from cache_long per-layer-trimmed to |prefix|.
    Generate N from each (using the token that would naturally follow prefix).
    Compare.
    """
    print("\n=== Phase 2 — per-layer-trim invariance (same source) ===", flush=True)

    # Control: fresh prefill only, decode from the post-prefix logits.
    cache_ctrl = _make_cache(lm)
    logits = _prefill(lm, prefix_ids, cache_ctrl)
    next_tok = int(mx.argmax(logits[0, -1, :]).item())
    ctrl = [next_tok]
    for _ in range(n_new - 1):
        a = mx.array([[next_tok]], dtype=mx.int32)
        logits = _forward_logits(lm, a, cache_ctrl)
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        ctrl.append(next_tok)

    # Build cache_long: prefill + filler_n generated tokens.
    cache_long = _make_cache(lm)
    logits = _prefill(lm, prefix_ids, cache_long)
    tok = int(mx.argmax(logits[0, -1, :]).item())
    for _ in range(filler_n):
        a = mx.array([[tok]], dtype=mx.int32)
        logits = _forward_logits(lm, a, cache_long)
        tok = int(mx.argmax(logits[0, -1, :]).item())

    # Snapshot at prefix depth (separate fresh cache, prefill only).
    cache_snap = _make_cache(lm)
    _prefill(lm, prefix_ids, cache_snap)

    sliceable, non_sliceable = _classify_layers(cache_long)
    print(f"  sliceable layers    : {len(sliceable)}", flush=True)
    print(f"  non-sliceable layers: {len(non_sliceable)}", flush=True)
    print(f"  layer class sample  : "
          f"{[type(c).__name__ for c in cache_long[:6]]}", flush=True)

    # Stitch.
    target_depth = len(prefix_ids)
    stitched = _stitch(cache_snap, cache_long, non_sliceable, sliceable, target_depth)
    print(f"  cache offsets post-stitch (first 6): "
          f"{_cache_offsets(stitched)[:6]}", flush=True)

    # Decode from stitched using the SAME post-prefix logits as control.
    # We need a starting token; the stitched cache has no pre-computed logits.
    # So we feed a single-token forward with an UNUSED sentinel?  No — we
    # instead start with the control's first token (ctrl[0]) and verify the
    # REMAINDER matches.  That's legit — ctrl[0] depends only on the post-
    # prefix logits which we don't need from the stitched cache.
    next_tok = ctrl[0]
    exp = [next_tok]
    for _ in range(n_new - 1):
        a = mx.array([[next_tok]], dtype=mx.int32)
        logits = _forward_logits(lm, a, stitched)
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        exp.append(next_tok)

    ok = ctrl == exp
    print(f"  control : {ctrl[:8]}{'...' if len(ctrl) > 8 else ''}", flush=True)
    print(f"  stitched: {exp[:8]}{'...' if len(exp) > 8 else ''}", flush=True)
    print(f"  result  : {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        for i, (a, b) in enumerate(zip(ctrl, exp)):
            if a != b:
                print(f"  first divergence at position {i}: ctrl={a} exp={b}", flush=True)
                break
    return ok


def phase3_cross_source(
    lm, prefix_ids: List[int], n_new: int = 16, filler_n: int = 32
) -> bool:
    """The real test: snapshot from session A (prefill X), sliceable from
    session B (prefill X + different_continuation).  Stitch.  Must match
    fresh prefill(X) + decode.

    To force session B's sliceable layers to be populated from X but with a
    cache-history that went past X into Y_b: we prefill X, then decode
    filler_n tokens (Y_b), then trim back to |X|.  The trimmed sliceable
    layers' tensor positions [0:|X|] should be bit-exact to what a fresh
    prefill would produce.  (The only question is numerical determinism.)
    """
    print("\n=== Phase 3 — cross-source stitch ===", flush=True)

    # Control: fresh prefill then decode.
    cache_ctrl = _make_cache(lm)
    logits = _prefill(lm, prefix_ids, cache_ctrl)
    next_tok = int(mx.argmax(logits[0, -1, :]).item())
    ctrl = [next_tok]
    for _ in range(n_new - 1):
        a = mx.array([[next_tok]], dtype=mx.int32)
        logits = _forward_logits(lm, a, cache_ctrl)
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        ctrl.append(next_tok)

    # Session A: snapshot (prefill X only).
    cache_A_snap = _make_cache(lm)
    _prefill(lm, prefix_ids, cache_A_snap)

    # Session B: prefill X then decode Y_b.  Force a DIFFERENT continuation
    # from session A by injecting a contrary seed token.  Since session A
    # didn't decode, "different" is mostly about KVCache positions [|X|:]
    # being filled; we don't care, we'll trim them off.
    cache_B_long = _make_cache(lm)
    logits = _prefill(lm, prefix_ids, cache_B_long)
    # Seed with a non-greedy token to make the trajectory differ from
    # anything A would produce.
    seed_tok = int(mx.argmax(logits[0, -1, :]).item())
    # Pick a genuinely different seed: 2nd-best token.
    vocab_logits = logits[0, -1, :]
    vocab_logits_np = vocab_logits.tolist()
    order = sorted(range(len(vocab_logits_np)), key=lambda i: -vocab_logits_np[i])
    alt_seed = order[1] if len(order) > 1 else seed_tok
    tok = alt_seed
    for _ in range(filler_n):
        a = mx.array([[tok]], dtype=mx.int32)
        logits = _forward_logits(lm, a, cache_B_long)
        tok = int(mx.argmax(logits[0, -1, :]).item())
    print(f"  session B generated {filler_n} tokens starting from alt_seed={alt_seed}",
          flush=True)

    # Stitch: non-sliceable from A_snap, sliceable from B_long trimmed to |X|.
    sliceable, non_sliceable = _classify_layers(cache_B_long)
    target_depth = len(prefix_ids)
    stitched = _stitch(cache_A_snap, cache_B_long, non_sliceable, sliceable, target_depth)

    # Decode from stitched, bootstrapped with ctrl[0] (control's first token).
    next_tok = ctrl[0]
    exp = [next_tok]
    for _ in range(n_new - 1):
        a = mx.array([[next_tok]], dtype=mx.int32)
        logits = _forward_logits(lm, a, stitched)
        next_tok = int(mx.argmax(logits[0, -1, :]).item())
        exp.append(next_tok)

    ok = ctrl == exp
    print(f"  control : {ctrl[:8]}{'...' if len(ctrl) > 8 else ''}", flush=True)
    print(f"  stitched: {exp[:8]}{'...' if len(exp) > 8 else ''}", flush=True)
    print(f"  result  : {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        for i, (a, b) in enumerate(zip(ctrl, exp)):
            if a != b:
                print(f"  first divergence at position {i}: ctrl={a} exp={b}", flush=True)
                break
    return ok


def main() -> int:
    model_path = os.environ.get("MODEL_PATH") or ""
    if not model_path:
        # Parse .env
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("MODEL_PATH=") and not line.startswith("#"):
                    model_path = line.split("=", 1)[1].strip()
                    break
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if not model_path:
        print("ERROR: no MODEL_PATH supplied (arg or .env)", file=sys.stderr)
        return 2

    lm, tokenizer = _load_model(model_path)
    prefix_ids = _tokenize_prefix(tokenizer, n_target=512)

    mx.clear_cache()
    p1 = phase1_deepcopy(lm, prefix_ids, n_new=16)
    mx.clear_cache()
    p2 = phase2_trim(lm, prefix_ids, n_new=16, filler_n=32)
    mx.clear_cache()
    p3 = phase3_cross_source(lm, prefix_ids, n_new=16, filler_n=32)

    print("\n=== Summary ===", flush=True)
    print(f"  Phase 1 (deepcopy invariance)     : {'PASS' if p1 else 'FAIL'}", flush=True)
    print(f"  Phase 2 (per-layer-trim invariance): {'PASS' if p2 else 'FAIL'}", flush=True)
    print(f"  Phase 3 (cross-source stitch)     : {'PASS' if p3 else 'FAIL'}", flush=True)
    print("", flush=True)
    if p1 and p2 and p3:
        print("  VERDICT: G4v2 stitching is SOUND.  Proceed with design.", flush=True)
        return 0
    if p1 and p2 and not p3:
        print("  VERDICT: stitch ORDERING issue — cross-source coupling.  Investigate.",
              flush=True)
        return 1
    if p1 and not p2:
        print("  VERDICT: per-layer-trim is BROKEN for this model.  "
              "ArraysCache path-dependent or trim primitive wrong.  "
              "Fall back to G4v1 whole-cache snapshots.", flush=True)
        return 1
    if not p1:
        print("  VERDICT: deepcopy is BROKEN.  Must mx.eval + manual copy.", flush=True)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
