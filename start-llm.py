import json
import time
import subprocess
import atexit
import copy
import os
import threading
import tempfile
import re
import uuid
import hashlib
import sys
import shutil
import signal
import math
from datetime import datetime
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer, HTTPServer
from dotenv import load_dotenv

# --- THE METAL CRASH FIX (CPU QUARANTINE) ---
import torch

# Blindfold PyTorch so it never touches the Apple Silicon GPU (MPS).
# This forces torchvision to use the CPU, preventing Metal Command Buffer
# collisions with MLX during massive OpenClaw prefills.
torch.backends.mps.is_built = lambda: False
# --------------------------------------------

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import (
    make_prompt_cache,
    can_trim_prompt_cache,
    trim_prompt_cache,
)

# Optional VLM support (Blaizzy/mlx-vlm). If unavailable, is_vlm is always False.
try:
    import mlx_vlm
    from mlx_vlm import load as load_vlm
    from mlx_vlm import stream_generate as stream_generate_vlm
    from mlx_vlm.utils import load_config as load_vlm_config
    from mlx_vlm.utils import prepare_inputs as vlm_prepare_inputs
    from mlx_vlm.utils import load_image as vlm_load_image
    from mlx_vlm.prompt_utils import get_chat_template

    mlx_vlm_available = True
except ImportError:
    mlx_vlm_available = False
    load_vlm = None
    stream_generate_vlm = None
    load_vlm_config = None
    vlm_prepare_inputs = None
    vlm_load_image = None
    get_chat_template = None

SCRIPT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = SCRIPT_DIR / ".env"

# Thread-local VLM diagnostics (used_prefix_stable, etc.) for cache-debug logging.
_vlm_diagnostics = threading.local()
if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None else value.strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _env_str_any(names: List[str], default: str) -> str:
    for name in names:
        raw = os.getenv(name)
        if raw is not None and raw.strip() != "":
            return raw.strip()
    return default


def _env_int_any(names: List[str], default: int) -> int:
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        try:
            return int(raw)
        except ValueError:
            continue
    return default


def _env_bool_any(names: List[str], default: bool) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is None or raw.strip() == "":
            continue
        normalized = raw.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


def _env_kv_bits(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().upper()
    if normalized == "OFF":
        return None
    try:
        return float(normalized)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model_path: str
    model_family: str
    mlx_host: str
    mlx_port: int
    proxy_port: int
    prompt_cache_max_entries_global: int
    prompt_cache_max_entries_per_session: int
    prompt_cache_ttl_seconds: int
    prompt_cache_session_max_idle_seconds: int
    max_kv_size: int
    kv_group_size: int
    kv_bits: Optional[float]
    kv_quant_scheme: str
    quantized_kv_start: int
    default_temperature: float
    default_top_p: float
    default_top_k: int
    default_min_p: float
    default_repetition_penalty: float
    default_repetition_context_size: int
    default_max_tokens: int
    enable_request_logging: bool
    default_thinking: bool
    vlm_cache_debug: bool
    cache_vlm_image_identity: bool
    normalize_write_tool_content_for_prompt: bool
    cache_canonicalize_tool_context: bool
    cache_session_partitioning: bool
    prompt_cache_block_size: int
    cache_use_block_index: bool
    cache_norm_safety_check: bool
    log_root: Path
    proxy_startup_wait_seconds: float
    proxy_model_id: str
    preserve_thinking: bool
    generation_watchdog_seconds: int


def _normalize_model_family(value: Optional[str]) -> str:
    raw = (value or "").strip().lower()
    if raw in {"qwen", "qwen3", "qen3"}:
        return "qwen3"
    if raw in {"glm", "glm4", "glm-4"}:
        return "glm4"
    return "generic"


def _infer_model_family(model_path: str) -> str:
    normalized = (model_path or "").strip().lower()
    if "qwen" in normalized:
        return "qwen3"
    if "glm" in normalized:
        return "glm4"
    return "generic"


def _build_settings() -> Settings:
    model_path = _env_str("MODEL_PATH", "lmstudio-community/GLM-4.7-Flash-MLX-4bit")
    model_family = _normalize_model_family(
        _env_str("MODEL_FAMILY", _infer_model_family(model_path))
    )
    proxy_model_id = _env_str(
        "PROXY_MODEL_ID",
        model_path if model_path.startswith("openai/") else f"openai/{model_path}",
    )
    return Settings(
        model_path=model_path,
        model_family=model_family,
        mlx_host=_env_str("MLX_HOST", "127.0.0.1"),
        mlx_port=_env_int("MLX_PORT", 8080),
        proxy_port=_env_int("PROXY_PORT", 4000),
        prompt_cache_max_entries_global=_env_int_any(
            ["PROMPT_CACHE_MAX_ENTRIES_GLOBAL", "PROMPT_CACHE_MAX_SIZE"],
            24,
        ),
        prompt_cache_max_entries_per_session=_env_int_any(
            ["PROMPT_CACHE_MAX_ENTRIES_PER_SESSION"],
            2,
        ),
        prompt_cache_ttl_seconds=_env_int("PROMPT_CACHE_TTL_SECONDS", 30 * 60),
        prompt_cache_session_max_idle_seconds=_env_int_any(
            ["PROMPT_CACHE_SESSION_MAX_IDLE_SECONDS"],
            30 * 60,
        ),
        max_kv_size=_env_int("MAX_KV_SIZE", 196608),
        kv_group_size=_env_int("KV_GROUP_SIZE", 64),
        kv_bits=_env_kv_bits("KV_BITS", None),
        kv_quant_scheme=_env_str("KV_QUANT_SCHEME", "uniform"),
        quantized_kv_start=_env_int("QUANTIZED_KV_START", 5000),
        default_temperature=_env_float("DEFAULT_TEMPERATURE", 0.1),
        default_top_p=_env_float("DEFAULT_TOP_P", 0.9),
        default_top_k=_env_int("DEFAULT_TOP_K", 40),
        default_min_p=_env_float("DEFAULT_MIN_P", 0.05),
        default_repetition_penalty=_env_float("DEFAULT_REPETITION_PENALTY", 1.08),
        default_repetition_context_size=_env_int(
            "DEFAULT_REPETITION_CONTEXT_SIZE", 256
        ),
        default_max_tokens=_env_int("DEFAULT_MAX_TOKENS", 2048),
        enable_request_logging=_env_bool("ENABLE_REQUEST_LOGGING", True),
        default_thinking=_env_bool("DEFAULT_THINKING", True),
        vlm_cache_debug=_env_bool("VLM_CACHE_DEBUG", False),
        cache_vlm_image_identity=_env_bool("CACHE_VLM_IMAGE_IDENTITY", True),
        normalize_write_tool_content_for_prompt=_env_bool(
            "NORMALIZE_WRITE_TOOL_CONTENT_FOR_PROMPT", False
        ),
        cache_canonicalize_tool_context=_env_bool_any(
            ["CACHE_CANONICALIZE_TOOL_CONTEXT"],
            True,
        ),
        cache_session_partitioning=_env_bool_any(
            ["CACHE_SESSION_PARTITIONING"],
            True,
        ),
        prompt_cache_block_size=_env_int("PROMPT_CACHE_BLOCK_SIZE", 16),
        cache_use_block_index=_env_bool_any(
            ["CACHE_USE_BLOCK_INDEX"],
            True,
        ),
        cache_norm_safety_check=_env_bool("CACHE_NORM_SAFETY_CHECK", False),
        log_root=Path(_env_str("LOG_ROOT", str(SCRIPT_DIR / "logs"))),
        proxy_startup_wait_seconds=_env_float("PROXY_STARTUP_WAIT_SECONDS", 2.0),
        proxy_model_id=proxy_model_id,
        preserve_thinking=_env_bool("PRESERVE_THINKING", True),
        generation_watchdog_seconds=_env_int("GENERATION_WATCHDOG_SECONDS", 600),
    )


SETTINGS = _build_settings()

# VLM model_type whitelist (from mlx-vlm 0.4.3 models/ directory).
VLM_MODEL_TYPES = frozenset(
    {
        "aya_vision",
        "deepseek_vl_v2",
        "deepseekocr",
        "deepseekocr_2",
        "dots_ocr",
        "ernie4_5_moe_vl",
        "falcon_ocr",
        "falcon_perception",
        "fastvlm",
        "florence2",
        "gemma3",
        "gemma3n",
        "gemma4",
        "glm4v",
        "glm4v_moe",
        "glm_ocr",
        "granite4_vision",
        "granite_vision",
        "hunyuan_vl",
        "idefics2",
        "idefics3",
        "internvl_chat",
        "jina_vlm",
        "kimi_vl",
        "lfm2_vl",
        "llama4",
        "llava",
        "llava_bunny",
        "llava_next",
        "minicpmo",
        "mistral3",
        "mistral4",
        "mllama",
        "molmo",
        "molmo2",
        "molmo_point",
        "moondream3",
        "multi_modality",
        "paddleocr_vl",
        "paligemma",
        "phi3_v",
        "phi4_siglip",
        "phi4mm",
        "pixtral",
        "qwen2_5_vl",
        "qwen2_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_omni_moe",
        "qwen3_vl",
        "qwen3_vl_moe",
        "rfdetr",
        "sam3",
        "sam3_1",
        "smolvlm",
    }
)


def _resolve_model_path_and_config():
    """Resolve MODEL_PATH to a local directory and load config.json. Returns (path, config dict or None)."""
    path_str = SETTINGS.model_path
    path = Path(path_str)
    if path.exists() and path.is_dir():
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    return path, json.load(f)
            except Exception:
                return path, None
        return path, None
    if mlx_vlm_available:
        try:
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(
                    repo_id=path_str,
                    allow_patterns=[
                        "*.json",
                        "*.safetensors",
                        "*.model",
                        "*.tiktoken",
                        "*.py",
                        "*.jinja",
                    ],
                )
            )
            config_path = path / "config.json"
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    return path, json.load(f)
            return path, None
        except Exception:
            pass
    return None, None


def _is_vlm_config(config: Optional[Dict[str, Any]]) -> bool:
    """True if config indicates a VLM (vision) model. Prefer exact model_type to avoid text-only (e.g. Qwen3-Coder) being misclassified."""
    if not config or not mlx_vlm_available:
        return False
    model_type = (config.get("model_type") or "").strip().lower()
    if model_type in VLM_MODEL_TYPES:
        return True
    vc = config.get("vision_config")
    if isinstance(vc, dict) and len(vc) > 0:
        return True
    archs = config.get("architectures") or []
    if any("VL" in str(a) or "Vision" in str(a) for a in archs):
        return True
    return False


# GLOBAL LOCK: Prevents concurrent GPU access (Fixes the crash)
model_lock = threading.Lock()
prompt_cache_lock = threading.Lock()
console_lock = threading.Lock()

proxy_process = None
proxy_config_path = None

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE
)
ARG_PAIR_PATTERN = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL | re.IGNORECASE,
)
QWEN_FUNCTION_PATTERN = re.compile(
    r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.DOTALL | re.IGNORECASE
)
QWEN_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE
)
INBOUND_META_MESSAGE_ID_PATTERN = re.compile(r'("message_id"\s*:\s*")[^"]+(")')
# NOTE: INBOUND_TRUSTED_CONTEXT_BLOCK_PATTERN, CACHE_STABLE_INBOUND_CONTEXT_BLOCK, and
# INBOUND_CONTEXT_TO_PROJECT_BOUNDARY_PATTERN were removed (Phase 7 Fix 2, 2026-03-14).
# The Inbound Context block is now handled structurally by _canonicalize_inbound_context_block()
# using plain string anchors with no DOTALL regex.  Do NOT re-add a DOTALL regex here —
# see Phase 5 in PLAN.md for why that approach caused a 17k-char lobotomy bug.
# OpenClaw sub-agent completion event: "Stats: runtime 1m52s • tokens 0 (in 0 / out 0)"
# The runtime duration is volatile (changes with each execution).  The UUIDs in
# session_key and session_id are frozen per injection (stable across turns), so only
# the runtime field needs to be normalised.  Single-line, structurally bounded — safe.
SUBAGENT_STATS_PATTERN = re.compile(r"(Stats: runtime\s+)[^\n•]+", re.IGNORECASE)
# Cache normalization: scrub volatile request-specific text so retries hit cache.
CACHE_TIME_PATTERN = re.compile(r"Current time is[^\n]+\.", re.IGNORECASE)
CACHE_CCH_PATTERN = re.compile(r"cch=[a-zA-Z0-9]+;?", re.IGNORECASE)
CACHE_BILLING_HEADER_PATTERN = re.compile(
    r"-anthropic-billing-header:\s*[a-zA-Z0-9\-]+", re.IGNORECASE
)
# Claude Code injects variable <system-reminder> text (e.g. "gentle reminder" vs "improve or augment");
# normalize so stream=true vs stream=false retries hit cache.
CACHE_SYSTEM_REMINDER_PATTERN = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL | re.IGNORECASE
)
# Strip reasoning from response content when returning to client (so reasoning is hidden).
# Full think blocks (<think>...</think>) and "orphan" </think> (reasoning with no opening tag, e.g. GLM-style).
THINK_TAG_STRIP_PATTERN = re.compile(
    r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE
)
# Orphan </think>: content from start up to and including </think> so we hide reasoning when model
# outputs "reasoning text</think>\n\nanswer" without a leading <think> tag.
THINK_ORPHAN_CLOSE_PATTERN = re.compile(r"^.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking_from_content(text: str) -> str:
    """Remove <think>...</think> blocks and leading content up to orphan </think> so reasoning is hidden."""
    if not isinstance(text, str):
        return text
    # First remove full think blocks.
    out = THINK_TAG_STRIP_PATTERN.sub("", text)
    # Then remove any leading reasoning that ends with </think> but has no <think> (e.g. GLM / Qwen VLM).
    out = THINK_ORPHAN_CLOSE_PATTERN.sub("", out, count=1)
    return out.strip()


# --- Streaming reasoning splitter ---
# Single source of truth for splitting raw model output into three disjoint
# streams: reasoning (goes to OpenAI `delta.reasoning_content`), content (goes
# to `delta.content`), and a tool-call tail buffered for end-of-stream
# extraction via `_extract_openai_tool_calls`.
#
# Why a state machine instead of post-hoc regex: the SSE path needs to emit
# deltas live during generation so clients (PI, OpenClaw, Claude) render
# thinking blocks and answer text as they arrive instead of receiving a single
# "answer dump" at end-of-stream.  A regex pass only works on the full string.
#
# Invariant (non-pathological cases — every `<think>` eventually closed):
#     "".join(content_chunks) == stripped text (modulo trailing whitespace),
#     where "stripped" means the same transform `_strip_thinking_from_content`
#     performs — <think>...</think> + trailing \s* removed, plus orphan
#     leading ...</think> handled for non-qwen3 thinking turns.
#
# Pathological case (thinking=True but the model never emits </think>): the
# entire output is routed to reasoning_content.  Legacy `_strip_thinking_from_content`
# would have left the prepended "<think>" prefix in `message.content` in this
# case; strictly better UX to route the stream to the reasoning channel.  The
# non-streaming path keeps the legacy transform for `message.content` (purely
# additive `reasoning_content` field) so no back-compat break.
_THINK_OPEN_TAG = "<think>"
_THINK_CLOSE_TAG = "</think>"
_TOOL_CALL_OPEN_TAG = "<tool_call>"
_STREAM_SENTINEL_LOOKBACK = (
    max(len(_THINK_OPEN_TAG), len(_THINK_CLOSE_TAG), len(_TOOL_CALL_OPEN_TAG)) - 1
)


class _ReasoningStreamSplitter:
    """
    Feed raw model text; get ('reasoning'|'content', str) chunks out.  Call
    `flush()` at end-of-stream for residual emission plus the raw tool-call
    buffer (for downstream `_extract_openai_tool_calls`).

    Lookback: holds up to `_STREAM_SENTINEL_LOOKBACK` trailing chars in the
    internal buffer so a `<think>` / `</think>` / `<tool_call>` tag split
    across two generated-text chunks is detected correctly.  Content deltas
    therefore lag the true generation by up to that many chars — acceptable
    (< one visible token for every model we run).

    Tool-call mode: entering on first `<tool_call>` sentinel stops all
    content/reasoning emission and buffers the rest of the stream for
    end-of-stream extraction.  Text that precedes the sentinel is still
    emitted as content.  Text that follows `</tool_call>` in the same stream
    is returned from `flush()` via `_extract_openai_tool_calls` cleaning and
    emitted as a final content delta in the caller (mirroring the
    non-streaming behaviour where trailing text after a tool call stays in
    `message.content`).
    """

    MODE_REASONING = "reasoning"
    MODE_CONTENT = "content"
    MODE_TOOL = "tool"

    def __init__(self, enable_thinking: bool, model_family: str):
        self._enable_thinking = bool(enable_thinking)
        self._model_family = model_family
        self._buf = ""
        # When thinking is requested, start in REASONING.  This mirrors
        # `_normalize_assistant_text` which prepends "<think>" for non-qwen3
        # (so the orphan-close case is handled naturally) and for qwen3
        # (where the chat template supplies the opening `<think>` inside the
        # rendered prompt, so the model's generated output begins mid-think
        # and emits `</think>` before the answer).
        self._mode = self.MODE_REASONING if self._enable_thinking else self.MODE_CONTENT
        # Consume a single leading "<think>" tag if present (Qwen3 variants
        # whose templates do not pre-open the think block).  Checked once.
        self._consumed_leading_think_tag = False
        # After consuming `</think>`, the `\s*` trail in
        # `_strip_thinking_from_content` eats any whitespace before the answer.
        # If that whitespace arrives in a LATER feed, we cannot consume it
        # in the same pass as the close tag — this flag defers the strip
        # into CONTENT mode's next iteration.
        self._pending_content_ws_strip = False
        self._tool_buffer = ""

    def feed(self, text: str) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        if not text:
            return out
        self._buf += text
        self._process(out, final=False)
        return out

    def flush(self) -> Tuple[List[Tuple[str, str]], str]:
        """End-of-stream drain.  Returns (chunks, tool_buffer)."""
        out: List[Tuple[str, str]] = []
        self._process(out, final=True)
        if self._buf:
            if self._mode == self.MODE_CONTENT:
                trimmed = self._buf.rstrip()
                if trimmed:
                    out.append(("content", trimmed))
            elif self._mode == self.MODE_REASONING:
                out.append(("reasoning", self._buf))
            # MODE_TOOL: already captured into self._tool_buffer.
            self._buf = ""
        tool_text = self._tool_buffer
        self._tool_buffer = ""
        return out, tool_text

    def _process(self, out: List[Tuple[str, str]], final: bool) -> None:
        while self._buf:
            if self._mode == self.MODE_TOOL:
                self._tool_buffer += self._buf
                self._buf = ""
                return
            if self._mode == self.MODE_REASONING:
                if not self._consumed_leading_think_tag:
                    stripped = self._buf.lstrip()
                    ws_len = len(self._buf) - len(stripped)
                    if stripped.startswith(_THINK_OPEN_TAG):
                        self._buf = self._buf[ws_len + len(_THINK_OPEN_TAG):]
                        self._consumed_leading_think_tag = True
                        continue
                    if len(stripped) >= len(_THINK_OPEN_TAG):
                        # Enough chars to rule out a leading <think>.
                        self._consumed_leading_think_tag = True
                    elif not final:
                        return  # Wait for more data.
                    else:
                        self._consumed_leading_think_tag = True
                close_idx = self._buf.find(_THINK_CLOSE_TAG)
                if close_idx == -1:
                    hold = len(_THINK_CLOSE_TAG) - 1 if not final else 0
                    emit_upto = max(0, len(self._buf) - hold)
                    if emit_upto > 0:
                        out.append(("reasoning", self._buf[:emit_upto]))
                        self._buf = self._buf[emit_upto:]
                    return
                if close_idx > 0:
                    out.append(("reasoning", self._buf[:close_idx]))
                consumed = close_idx + len(_THINK_CLOSE_TAG)
                # Consume \s* after </think> to match _strip_thinking_from_content.
                while consumed < len(self._buf) and self._buf[consumed] in " \t\n\r":
                    consumed += 1
                # If we ran out of buffer while still on whitespace territory,
                # finish the \s* strip on the next feed in CONTENT mode.
                self._pending_content_ws_strip = consumed >= len(self._buf)
                self._buf = self._buf[consumed:]
                self._mode = self.MODE_CONTENT
                continue
            # MODE_CONTENT
            if self._pending_content_ws_strip and self._buf:
                stripped = self._buf.lstrip()
                ws_count = len(self._buf) - len(stripped)
                if ws_count > 0:
                    self._buf = stripped
                if stripped:
                    # First non-whitespace char seen — done stripping.
                    self._pending_content_ws_strip = False
                elif not final:
                    # Still pure whitespace and more data may come; wait.
                    return
            open_think_idx = self._buf.find(_THINK_OPEN_TAG)
            open_tool_idx = self._buf.find(_TOOL_CALL_OPEN_TAG)
            candidates = [i for i in (open_think_idx, open_tool_idx) if i != -1]
            if candidates:
                sentinel_idx = min(candidates)
                if sentinel_idx > 0:
                    out.append(("content", self._buf[:sentinel_idx]))
                if sentinel_idx == open_think_idx and (
                    open_tool_idx == -1 or open_think_idx <= open_tool_idx
                ):
                    self._buf = self._buf[sentinel_idx + len(_THINK_OPEN_TAG):]
                    self._mode = self.MODE_REASONING
                else:
                    # Tool-call sentinel: preserve the tag itself in the buffer
                    # so `_extract_openai_tool_calls` can find it end-of-stream.
                    self._tool_buffer += self._buf[sentinel_idx:]
                    self._buf = ""
                    self._mode = self.MODE_TOOL
                continue
            hold = _STREAM_SENTINEL_LOOKBACK if not final else 0
            emit_upto = max(0, len(self._buf) - hold)
            if emit_upto > 0:
                chunk = self._buf[:emit_upto]
                if final:
                    # Mirror `_strip_thinking_from_content`'s trailing
                    # `.strip()` so the concatenation of content deltas the
                    # client sees is byte-identical to what the non-streaming
                    # path returns in `message.content`.  Healing-store
                    # lookups otherwise diverge when a client retries a
                    # stream=true request as stream=false (or vice versa).
                    chunk = chunk.rstrip()
                if chunk:
                    out.append(("content", chunk))
                self._buf = self._buf[emit_upto:]
            return


def _split_text_for_reasoning(
    raw_text: str,
    enable_thinking: bool,
    model_family: str,
) -> Tuple[str, str]:
    """
    Apply the streaming splitter to a complete raw_text buffer for the
    non-streaming path.  Returns `(content, reasoning)`.  Tool-call extraction
    is left to the caller (consistent with the streaming path).

    Used to produce the `message.reasoning_content` field on non-streaming
    responses additively — `message.content` remains derived from
    `_strip_thinking_from_content` so any client that does not consume
    `reasoning_content` sees byte-identical legacy content.
    """
    if not isinstance(raw_text, str) or not raw_text:
        return "", ""
    splitter = _ReasoningStreamSplitter(enable_thinking, model_family)
    chunks = splitter.feed(raw_text)
    tail, _tool_text = splitter.flush()
    chunks.extend(tail)
    content = "".join(c for k, c in chunks if k == "content")
    reasoning = "".join(c for k, c in chunks if k == "reasoning")
    return content, reasoning


def _selftest_reasoning_splitter() -> None:
    """
    Import-time invariant check for `_ReasoningStreamSplitter`.  Runs once at
    startup; raises AssertionError on regression so the server refuses to
    start rather than silently shipping a broken reasoning split.  Keeping it
    here (no external test harness in this repo) preserves rule 9's
    "evidence before code" discipline for a state machine that is otherwise
    easy to break with a lookback off-by-one.
    """
    cases = [
        # (raw, enable_thinking, model_family, expected_content, expected_reasoning, expected_tool)
        ("<think>r</think>\nans", True, "qwen3", "ans", "r", ""),
        ("r</think>\nans", True, "glm4", "ans", "r", ""),
        ("no think", False, "qwen3", "no think", "", ""),
        (
            "<think>r1</think>\n\nans <think>r2</think> more",
            True, "qwen3", "ans more", "r1r2", "",
        ),
        (
            "reasoning</think>\n<tool_call><function=foo></function></tool_call>",
            True, "glm4", "", "reasoning",
            "<tool_call><function=foo></function></tool_call>",
        ),
        (
            "<tool_call>x</tool_call>", False, "qwen3",
            "", "", "<tool_call>x</tool_call>",
        ),
        ("unclosed thinking", True, "glm4", "", "unclosed thinking", ""),
        # Trailing whitespace — splitter output must match _strip_thinking_from_content.
        ("<think>r</think>\nans\n", True, "qwen3", "ans", "r", ""),
        ("<think>r</think>\n  answer  ", True, "qwen3", "answer", "r", ""),
        # Content already in stream at flush with no closing newline.
        ("r</think>answer", True, "glm4", "answer", "r", ""),
    ]
    for raw, et, mf, exp_c, exp_r, exp_t in cases:
        # Chunk size 3 to exercise tag-split lookback.
        splitter = _ReasoningStreamSplitter(et, mf)
        collected: List[Tuple[str, str]] = []
        for i in range(0, len(raw), 3):
            collected.extend(splitter.feed(raw[i:i + 3]))
        tail, tool_text = splitter.flush()
        collected.extend(tail)
        got_c = "".join(c for k, c in collected if k == "content")
        got_r = "".join(c for k, c in collected if k == "reasoning")
        assert got_c == exp_c, (
            f"splitter content mismatch raw={raw!r} got={got_c!r} exp={exp_c!r}"
        )
        assert got_r == exp_r, (
            f"splitter reasoning mismatch raw={raw!r} got={got_r!r} exp={exp_r!r}"
        )
        assert tool_text == exp_t, (
            f"splitter tool mismatch raw={raw!r} got={tool_text!r} exp={exp_t!r}"
        )

    # Additional invariant: for any CLOSED-thinking case with NO tool call,
    # the stream-view content must equal
    # `_strip_thinking_from_content(_normalize_assistant_text(raw))` so
    # streaming and non-streaming content stay byte-identical (healing-
    # hash integrity).  Unclosed-thinking (pathological) and tool-call cases
    # are excluded — see class docstring for why.
    #
    # `_normalize_assistant_text` is defined later in this file; the caller
    # at module load time (see `_selftest_reasoning_splitter_run_after_deps`
    # below) schedules this check to run AFTER that definition.  Falling
    # back to an inline shim here keeps the selftest self-contained if
    # someone invokes it in isolation.
    def _legacy_normalize(text: str, et: bool, mf: str) -> str:
        normalize = globals().get("_normalize_assistant_text")
        if normalize is not None:
            return normalize(text, et, mf)
        if mf == "qwen3":
            return text
        if et and text and not text.lstrip().startswith("<think>"):
            return "<think>" + text
        return text

    for raw, et, mf, _exp_c, _exp_r, exp_t in cases:
        if exp_t:
            continue
        if et and "</think>" not in raw:
            continue  # pathological: documented divergence
        splitter = _ReasoningStreamSplitter(et, mf)
        chunks = splitter.feed(raw)
        tail, _ = splitter.flush()
        chunks.extend(tail)
        got_c = "".join(c for k, c in chunks if k == "content")
        if et:
            legacy = _strip_thinking_from_content(_legacy_normalize(raw, et, mf))
        else:
            legacy = raw  # thinking off → no stripping on non-streaming
        assert got_c == legacy, (
            f"splitter/legacy divergence raw={raw!r} et={et} mf={mf} "
            f"splitter={got_c!r} legacy={legacy!r}"
        )


# Run the splitter-only part of the selftest immediately (no forward
# references).  The equivalence subtest inside `_selftest_reasoning_splitter`
# looks up `_normalize_assistant_text` dynamically via globals() so it is safe
# to call here even though the real function is defined further down — it
# falls back to an inline shim.  This keeps import-time failure loud if the
# splitter regresses.
_selftest_reasoning_splitter()


# --- STATELESS MESSAGE HEALING STORE ---
HEALING_STORE: OrderedDict = OrderedDict()
HEALING_STORE_LOCK = threading.Lock()
MAX_HEALING_STORE = 2000  # Generous size to survive deep multi-agent sessions


# --- P4: Cache-correctness observability ---
# Three counters that must stay at 0 on clean traffic.  Any bump is a signal
# that an assumption from P1-P3 has slipped.  See LESSONS.md "Cache-
# Correctness Metrics" and .context/TASKS.md P4 row for interpretation.
#
#  vlm_retreat                    — G1 failed to prevent a mid-image cache cut
#                                   (canonical key collided with a different
#                                   image expansion).  Expected: 0.
#  hybrid_trim_miss               — `_trim_to_model_depth` rejected reuse
#                                   because `can_trim_prompt_cache` was False
#                                   on a hybrid VLM cache.  Nonzero justifies
#                                   G4 (pre-generation prompt-only checkpoint).
#  exact_key_rejected_by_model_lcp — exact canonical-key hit rejected because
#                                   `entry.model_tokens` was not a prefix of
#                                   the new request's model_tokens.  Nonzero
#                                   indicates a canonical-key derivation bug
#                                   (two distinct model streams producing the
#                                   same canonical key); tracked under G6.
_CACHE_METRICS: Dict[str, int] = {
    "vlm_retreat": 0,
    "hybrid_trim_miss": 0,
    "exact_key_rejected_by_model_lcp": 0,
}
_CACHE_METRICS_LOCK = threading.Lock()
# Per-request tracking: which counters bumped during THIS request, keyed by
# request_id.  Cleaned on request completion.  Enables the request log to
# report `cache_metrics_bumped_this_request` without the request having to
# thread state through the counter callers.
_CACHE_METRICS_PER_REQUEST: Dict[str, List[str]] = {}


def _bump_metric(name: str, request_id: Optional[str] = None, by: int = 1) -> int:
    """
    Atomically bump a cache-correctness counter.  Returns the new total.
    Also emits a one-line terminal status (`🧪 metric <name> bumped ...`)
    so an operator sees the event immediately — bumps are rare by design,
    so the noise is low and signal is high.
    """
    with _CACHE_METRICS_LOCK:
        if name not in _CACHE_METRICS:
            _CACHE_METRICS[name] = 0
        _CACHE_METRICS[name] += by
        new_total = _CACHE_METRICS[name]
        if request_id:
            bucket = _CACHE_METRICS_PER_REQUEST.setdefault(request_id, [])
            if name not in bucket:
                bucket.append(name)
    # Emit OUTSIDE the lock: _terminal_status takes console_lock, so keeping
    # the two lock regions disjoint removes any future ordering hazard if
    # console code ever needs to read metrics.
    _terminal_status(
        "🧪",
        f"metric {name} bumped (+{by}, total={new_total})"
        + (f" request={request_id}" if request_id else ""),
        indent=1,
    )
    return new_total


def _metrics_snapshot() -> Dict[str, int]:
    with _CACHE_METRICS_LOCK:
        return dict(_CACHE_METRICS)


def _metrics_drain_request(request_id: str) -> List[str]:
    """Pop and return the list of counters that bumped for `request_id`."""
    with _CACHE_METRICS_LOCK:
        return _CACHE_METRICS_PER_REQUEST.pop(request_id, [])


# Thread-local request id — used so the cache-layer bump sites (inside
# LRUPromptCache methods) can attribute to the request without having
# request_id threaded through every call.  Safe under the single-threaded
# HTTPServer (all requests share a single stack; set at do_POST entry and
# cleared in finally).  Under any future threaded server this would need
# revisiting; the server-level comment at run() already calls that out.
_current_request_id_tls = threading.local()


def _set_current_request_id(request_id: Optional[str]) -> None:
    _current_request_id_tls.value = request_id


def _current_request_id() -> Optional[str]:
    return getattr(_current_request_id_tls, "value", None)


def _canonicalize_tool_calls_for_hash(
    tool_calls: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Normalize tool_calls for stable hashing across clients that serialize
    `function.arguments` differently.  The server emits arguments as a
    JSON-STRING (per OpenAI spec); clients like pi echo them back as a
    native DICT.  Without canonicalization the healing hash diverges and
    cache hits are lost on every tool-call turn.

    Canonical form: arguments is ALWAYS a parsed mapping (dict) when the
    string is valid JSON; otherwise a raw-string placeholder.  The `id`
    field is kept (OpenAI uses it to bind tool results).
    """
    if not tool_calls:
        return None
    out: List[Dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            out.append(tc)
            continue
        tc_copy = dict(tc)
        fn = tc_copy.get("function")
        if isinstance(fn, dict):
            fn_copy = dict(fn)
            args = fn_copy.get("arguments")
            if isinstance(args, str):
                try:
                    fn_copy["arguments"] = json.loads(args)
                except Exception:
                    fn_copy["arguments"] = {"__raw__": args}
            tc_copy["function"] = fn_copy
        out.append(tc_copy)
    return out


def _get_healing_hash(
    text: str, tool_calls: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """
    Creates a robust SHA-256 hash of the assistant's output.
    If the text is empty but has tool calls, we hash the canonicalized tool calls.
    """
    base = (text or "").strip()
    canon = _canonicalize_tool_calls_for_hash(tool_calls)
    if canon:
        try:
            base += json.dumps(canon, sort_keys=True, ensure_ascii=False)
        except Exception:
            pass

    if not base:
        return None

    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _reasoning_already_embedded_in_content(
    reasoning_content: Any, content: Any
) -> bool:
    """
    True only when `content` already carries `reasoning_content` immediately
    followed by `</think>` — i.e. the flattening pattern clients like pi use
    to round-trip reasoning streams through OpenAI-style `messages[*].content`.

    Strict check by design: a substring match of just `</think>` would misfire
    on prompts that legitimately contain that literal (LLM-tooling discussions,
    template docs, chat logs of other agents).  Requiring the exact
    `reasoning_content + "</think>"` adjacency matches pi's echo and any other
    client using the same flattening, while leaving unrelated content alone.
    """
    if not isinstance(reasoning_content, str) or not reasoning_content:
        return False
    marker = reasoning_content + "</think>"

    def _text_carries_marker(text: Any) -> bool:
        return isinstance(text, str) and marker in text

    if _text_carries_marker(content):
        return True
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text") or part.get("content") or ""
                if _text_carries_marker(text):
                    return True
    return False


def _heal_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Intercepts incoming messages. If a stripped assistant message matches
    a hash in our store, we swap it back to the full version (with <think>).

    Also collapses pi-style echoes that set BOTH `content` (with thinking +
    `</think>` embedded) AND `reasoning_content`: the Qwen3.6 chat template
    with `preserve_thinking=True` uses `reasoning_content` to inject a
    `<think>…</think>` wrapper AND renders `content` verbatim — so any client
    that also puts the thinking inside `content` doubles the block, shifts
    every subsequent token, and turns the cache into a cold miss on every
    turn.  Verified against `chat_template.jinja`: when `reasoning_content`
    is absent, the template's preserve_thinking pass extracts the thinking
    from `content` at `</think>` and produces a byte-identical render to the
    model's original generation — so preserving the flag (and therefore the
    prior-reasoning context for the model) is safe; we only drop the
    redundant second copy.

    The guard is strict on purpose: we drop `reasoning_content` only when
    the SAME string literally appears inside `content` immediately before
    `</think>`.  That signature uniquely identifies the flattening pi
    performs when it round-trips the server's reasoning-aware stream
    through an OpenAI-shaped `messages` payload.  The common
    stripped-content-plus-reasoning_content case (legitimate supplementary
    reasoning) is left untouched; pathological content that contains a
    literal `</think>` without the matching reasoning prefix is also left
    untouched — at worst one cache miss, never a silent-correctness loss.
    """
    healed = []
    for msg in messages:
        m = dict(msg)
        if (m.get("role") or "").strip().lower() == "assistant":
            content = m.get("content", "")
            tool_calls = m.get("tool_calls")

            # Handle standard text content
            if isinstance(content, str):
                h = _get_healing_hash(content, tool_calls)
                if h:
                    with HEALING_STORE_LOCK:
                        if h in HEALING_STORE:
                            m["content"] = HEALING_STORE[h]
                            # CRITICAL: Prevent the Jinja template from double-rendering
                            # the tool calls, since the raw text already contains them.
                            m.pop("tool_calls", None)
                            HEALING_STORE.move_to_end(h, last=True)

            # Handle VLM list content (e.g., [{"type": "text", "text": "..."}])
            elif isinstance(content, list):
                new_content = []
                healed_any = False
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_part = part.get("text") or part.get("content") or ""
                        h = _get_healing_hash(text_part, tool_calls)
                        if h:
                            with HEALING_STORE_LOCK:
                                if h in HEALING_STORE:
                                    new_content.append(
                                        {**part, "text": HEALING_STORE[h]}
                                    )
                                    healed_any = True
                                    HEALING_STORE.move_to_end(h, last=True)
                                    continue
                    new_content.append(part)
                m["content"] = new_content
                if healed_any:
                    # CRITICAL: Prevent double-rendering for VLM multi-part messages too
                    m.pop("tool_calls", None)

            # Drop the redundant `reasoning_content` copy when the same
            # thinking text is already embedded inside `content`, preceding
            # `</think>`.  preserve_thinking still takes effect (the template
            # extracts the block from content), so the model keeps its prior
            # reasoning; only the duplicate render is removed.  See docstring
            # above for the safety argument behind the strict-prefix guard.
            if _reasoning_already_embedded_in_content(
                m.get("reasoning_content"), m.get("content")
            ):
                m.pop("reasoning_content", None)

        healed.append(m)
    return healed


# ---------------------------------------


def _terminal_status(icon: str, message: str, indent: int = 0) -> None:
    with console_lock:
        ts = datetime.now().strftime("%H:%M:%S")
        pad = "  " * max(indent, 0)
        print(f"{pad}{icon} [{ts}] {message}", flush=True)


def _debug_token_divergence(
    tokenizer,
    current_tokens: List[int],
    stored_tokens: Tuple[int, ...],
    context_window: int = 5,
):
    """Finds and prints exactly where two token sequences diverge for cache debugging."""
    min_len = min(len(current_tokens), len(stored_tokens))
    diverge_idx = -1

    for i in range(min_len):
        if current_tokens[i] != stored_tokens[i]:
            diverge_idx = i
            break

    # Limit how much we print: last few token IDs and short decoded snippets only
    max_tokens_show = 10
    max_text_len = 200

    start_idx = max(0, diverge_idx - context_window)
    tokens_before = current_tokens[start_idx:diverge_idx]
    if len(tokens_before) > max_tokens_show:
        tokens_before = tokens_before[-max_tokens_show:]
    end_idx_current = min(len(current_tokens), diverge_idx + context_window + 1)
    end_idx_stored = min(len(stored_tokens), diverge_idx + context_window + 1)

    print(f"\n" + "=" * 50)
    print(f"🚨 CACHE DIVERGENCE DETECTED AT INDEX {diverge_idx} 🚨")
    print(f"Token IDs before divergence (last {len(tokens_before)}): {tokens_before}")

    try:
        matching_text = tokenizer.decode(current_tokens[start_idx:diverge_idx])
        if len(matching_text) > max_text_len:
            matching_text = "..." + matching_text[-max_text_len:].strip()
        print(f"Matching text leading up: {repr(matching_text)}")

        curr_divergent_token = current_tokens[diverge_idx]
        stor_divergent_token = stored_tokens[diverge_idx]
        print(
            f"\n❌ Current Request Token [{diverge_idx}]: ID {curr_divergent_token} -> {repr(tokenizer.decode([curr_divergent_token]))}"
        )
        print(
            f"❌ Stored Cache Token  [{diverge_idx}]: ID {stor_divergent_token} -> {repr(tokenizer.decode([stor_divergent_token]))}"
        )

        curr_context_after = tokenizer.decode(
            current_tokens[diverge_idx + 1 : end_idx_current]
        )
        stor_context_after = tokenizer.decode(
            stored_tokens[diverge_idx + 1 : end_idx_stored]
        )
        if len(curr_context_after) > max_text_len:
            curr_context_after = curr_context_after[:max_text_len] + "..."
        if len(stor_context_after) > max_text_len:
            stor_context_after = stor_context_after[:max_text_len] + "..."
        print(f"\nCurrent context after: {repr(curr_context_after)}")
        print(f"Stored context after:  {repr(stor_context_after)}")
    except Exception as e:
        print(f"Could not decode tokens: {e}")
    print("=" * 50 + "\n")


def _block_chain_hashes(
    tokens: Tuple[int, ...],
    block_size: int,
) -> List[Tuple[bytes, int]]:
    """Return [(chain_hash, prefix_len), ...] for each block prefix. chain_hash[i] = H(prev || block_i)."""
    if block_size <= 0 or not tokens:
        return []
    out: List[Tuple[bytes, int]] = []
    prev = b""
    for i in range(0, len(tokens), block_size):
        block = tokens[i : i + block_size]
        block_bytes = b"".join(t.to_bytes(4, "big") for t in block)
        h = hashlib.sha256(prev + block_bytes).digest()
        prefix_len = min(i + block_size, len(tokens))
        out.append((h, prefix_len))
        prev = h
    return out


class LRUPromptCache:
    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        tokens: Tuple[int, ...]  # canonical cache_key = canonical prompt + generated
        model_tokens: Tuple[int, ...]  # model-space tokens used to BUILD the KV state
        count: int
        touched_at: float

    def __init__(self, max_size=10, ttl_seconds=1800):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.block_size = max(16, getattr(SETTINGS, "prompt_cache_block_size", 16))

        # Core Flat Cache: (model, exact_tokens) -> CacheEntry
        self._entries: Dict[Tuple[str, Tuple[int, ...]], self.CacheEntry] = {}
        # Block Hash Index: (model, chain_hash) -> Set of token sequences
        self._block_index: Dict[Tuple[str, bytes], set] = {}

    def _is_expired(self, entry):
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - entry.touched_at) > self.ttl_seconds

    def prune_expired(self):
        now = time.time()
        stale_keys = [k for k, v in self._entries.items() if self._is_expired(v)]
        for k in stale_keys:
            self._delete(k[0], k[1])

    def _delete(self, model, tokens):
        key = (model, tuple(tokens))
        if key not in self._entries:
            return

        # Clean up the block index to prevent memory leaks
        chain_pairs = _block_chain_hashes(tokens, self.block_size)
        for chain_hash, _ in chain_pairs:
            idx_key = (model, chain_hash)
            if idx_key in self._block_index:
                self._block_index[idx_key].discard(key[1])
                if not self._block_index[idx_key]:
                    del self._block_index[idx_key]

        del self._entries[key]

        # Return evicted Metal GPU buffers to the OS pool immediately.
        # Without this, MLX holds the backing Metal buffers in its pool even
        # after the Python reference is dropped, causing monotonic GPU memory
        # growth during long sessions with repeated cache divergence.
        try:
            mx.metal.clear_cache()
        except Exception:
            pass

    def _extract(self, model, tokens):
        key = (model, tuple(tokens))
        entry = self._entries[key]
        entry.touched_at = time.time()
        entry.count += 1  # Track hit frequency for eviction weighting

        return self.CacheEntry(
            copy.deepcopy(entry.prompt_cache),
            entry.tokens,
            entry.model_tokens,
            entry.count,
            entry.touched_at,
        )

    def _evict_optimal(self):
        """
        Cost-Aware Eviction: Finds the entry with the highest eviction score.
        Protects long 'trunks' and frequently used templates; penalizes old, short branches.
        """
        now = time.time()
        best_key = None
        max_score = -1.0

        for key, entry in self._entries.items():
            age_seconds = max(1.0, now - entry.touched_at)

            # Use square root to create a balanced gravity for long chains.
            # A 10,000 token chain has 10x more protection than a 100 token chain.
            length_weight = math.sqrt(len(entry.tokens))
            freq_weight = math.log1p(entry.count) + 1.0

            # Higher score = more likely to be evicted. (Old and Short -> High Score)
            score = age_seconds / (length_weight * freq_weight)

            if score > max_score:
                max_score = score
                best_key = key

        if best_key:
            self._delete(best_key[0], best_key[1])

    def _cull_redundant_prefixes(self, model, new_tokens, new_model_tokens=None):
        """
        Frees up slots by removing strict prefixes.  Since the cache can trim
        longer chains to serve shorter ones, shorter strict prefixes are
        wasted slots — when both canonical AND model tokens line up.

        P6/1d: For VLM the canonical-prefix relation alone is insufficient.
        Two distinct entries can share a canonical prefix (say up to the
        system-prompt / image marker boundary) while their `model_tokens`
        diverge because of different image expansions or per-client token
        drift.  Culling the shorter of two such entries silently discards
        a genuinely distinct KV state.  If `new_model_tokens` is provided,
        also require `new_model_tokens[:len(entry.model_tokens)] ==
        entry.model_tokens` before culling.  Callers that don't supply
        `new_model_tokens` (none in the current code, but kept for
        back-compat) fall back to the pre-P6 canonical-only check.

        With G1's image-identity markers, canonical collisions across
        distinct images are already prevented at the derivation layer;
        this guard is belt-and-braces for any future normalisation pass
        that might reintroduce cross-model-stream canonical aliasing.
        """
        new_len = len(new_tokens)
        to_delete = []
        for (m, t_tup), entry in self._entries.items():
            if m != model or len(t_tup) >= new_len:
                continue
            if new_tokens[: len(t_tup)] != t_tup:
                continue
            if new_model_tokens is not None:
                entry_model = entry.model_tokens
                if len(entry_model) > len(new_model_tokens):
                    continue
                if tuple(new_model_tokens[: len(entry_model)]) != tuple(entry_model):
                    # Same canonical prefix, divergent model stream — do
                    # NOT cull; this is a genuinely distinct KV state.
                    continue
            to_delete.append((m, t_tup))

        to_delete.sort(key=lambda x: len(x[1]), reverse=True)
        if to_delete:
            # Spare the longest prefix from deletion
            to_delete.pop(0)

        for k in to_delete:
            self._delete(k[0], k[1])

    @staticmethod
    def _lcp(a: Tuple[int, ...], b: Tuple[int, ...], limit: Optional[int] = None) -> int:
        """Longest common prefix length between two integer token sequences."""
        n = min(len(a), len(b))
        if limit is not None:
            n = min(n, limit)
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def fetch_nearest_cache(self, model, tokens, request_model_tokens=None):
        """
        Lookup a cached KV state that can serve `tokens` (canonical request
        cache key).  Uses canonical block hashes for candidate discovery, then
        verifies match in MODEL-token space using `request_model_tokens` —
        this guards against VLM cross-request leakage where two requests
        share a canonical prefix but diverge in image-pad expansion.

        Returns a tuple
            (prompt_cache, rest_model_tokens, selected_canonical_tokens,
             match_type, canonical_matched_prefix_len)
        where the returned `prompt_cache` has already been trimmed to the
        correct model-space KV depth (use `_kv_cache_offset(prompt_cache)` to
        read it back).  `rest_model_tokens` is the remaining tail of the new
        request's model_tokens that still needs prefilling.

        For text-only (LM) paths, canonical == model; callers can pass
        `request_model_tokens=None` and the routine falls back to the
        canonical tokens (preserving prior behaviour).
        """
        self.prune_expired()
        tokens_tup = tuple(tokens)
        req_model = (
            tuple(request_model_tokens)
            if request_model_tokens is not None
            else tokens_tup
        )

        def _trim_to_model_depth(cache_state, entry_model_tokens, target_model_len):
            """
            Align cache state with `target_model_len` in model space.
            Returns (ok, effective_depth) where `ok` signals whether the
            cache is usable and `effective_depth` is the KV depth the caller
            should treat as 'how many model tokens are already prefilled'.

            Hybrid VLM caches (Qwen3.5_moe = ArraysCache + KVCache) are not
            trimmable: `can_trim_prompt_cache` returns False because linear-
            attention layers have no position-wise trim semantics.  In that
            case we can still reuse the cache IFF the stored entry's full
            sequence is a strict prefix of the new request's model tokens —
            i.e. `target_model_len >= len(entry.model_tokens)` so no trim is
            actually required.  If the new request diverges mid-way and we
            cannot truncate, we report unusable (miss) rather than silently
            letting stale KV past the divergence point contaminate attention.
            """
            current_kv = _kv_cache_offset(cache_state)
            if current_kv is None:
                current_kv = len(entry_model_tokens)
            if target_model_len > current_kv:
                # Target deeper than stored — physically impossible.
                return False, current_kv
            trim_amount = current_kv - target_model_len
            if trim_amount == 0:
                return True, current_kv
            if can_trim_prompt_cache(cache_state):
                trim_prompt_cache(cache_state, trim_amount)
                return True, target_model_len
            # Non-trimmable hybrid cache.  Stored KV is at `current_kv`;
            # request has `target_model_len` safe tokens.  current_kv >
            # target_model_len means the stored sequence has tokens past
            # the common prefix — those KV positions hold attention from
            # a diverging continuation and would poison the new request.
            # Refuse cache reuse in this case (caller treats as miss).
            # P4: track the hybrid-trim-miss rate.  Each bump is a cache
            # opportunity lost because Qwen3_5Moe's ArraysCache layers
            # aren't trimmable.  Nonzero rate in production is what
            # justifies G4 (pre-generation prompt-only checkpoint) —
            # without that signal, G4 would be a speculative refactor.
            _bump_metric("hybrid_trim_miss", _current_request_id())
            return False, current_kv

        # Fast path: exact canonical-token key lookup.
        exact_key = (model, tokens_tup)
        if exact_key in self._entries:
            entry = self._extract(model, tokens_tup)
            # In model space: the entry's model_tokens must be a prefix of
            # the new request's model_tokens (or equal).  If not, the entry
            # was built from a different model sequence sharing only a
            # canonical prefix — unusable.
            model_lcp = self._lcp(entry.model_tokens, req_model)
            if model_lcp < len(entry.model_tokens):
                # Not a safe exact reuse; try the block-hash path below.
                # P4: nonzero rate here means two distinct model_token
                # streams produced the SAME canonical cache key — a bug
                # in `_canonicalize_messages` or `_scrub_cache_key` that
                # over-normalised and merged semantically-different
                # prompts (tracked under G6).  Should be 0 on clean paths.
                _bump_metric(
                    "exact_key_rejected_by_model_lcp", _current_request_id()
                )
            else:
                # Full reuse: trim down to (target_model_len - 1) so the
                # model recomputes exactly one token to kick off generation.
                target = max(1, len(req_model) - 1)
                ok, eff = _trim_to_model_depth(
                    entry.prompt_cache, entry.model_tokens, target
                )
                if not ok:
                    return None, list(req_model), tokens_tup, "miss", 0
                rest = list(req_model)[eff:]
                # P6/1c: report canonical match length WITHOUT the -1 model-
                # space bootstrap adjustment.  `matched_prefix_len` is a
                # canonical-space telemetry/M3 value; keeping it coherent
                # with the block-hash path avoids a spurious off-by-one
                # when the exact-key fast path fires instead of the block
                # walk (e.g. on repeated identical requests).
                return (
                    entry.prompt_cache,
                    rest,
                    tokens_tup,
                    "exact",
                    len(tokens_tup),
                )

        chain_pairs = _block_chain_hashes(tokens_tup, self.block_size)
        if not chain_pairs:
            return None, list(req_model), tokens_tup, "miss", 0

        best_canon_prefix_len = 0
        best_cached_tokens = None
        best_entry_model_tokens: Tuple[int, ...] = ()
        best_model_lcp = 0

        # Walk blocks backwards, collect candidates at deepest canonical
        # block boundary.  Among candidates at that level, pick the one with
        # LONGEST model-space LCP — that's the safest cache reuse for the
        # actual sequence being prefilled.
        for chain_hash, req_prefix_len in reversed(chain_pairs):
            idx_key = (model, chain_hash)
            if idx_key not in self._block_index:
                continue
            candidate_tokens_set = self._block_index[idx_key]
            # Prefer candidates whose stored model_tokens are fully covered
            # by the new request's model_tokens (no KV trim required).
            # Hybrid VLM caches (ArraysCache + KVCache) cannot be trimmed —
            # picking an entry that needs trimming would force a miss even
            # when a shorter "prompt-only" checkpoint would have been a
            # clean hit.  So rank candidates by (full_cover first, then
            # longer model_lcp), not by raw model_lcp alone.
            best_at_level = None
            best_at_level_full_cover = False
            best_at_level_model_lcp = -1
            best_at_level_entry_model: Tuple[int, ...] = ()
            for candidate_tokens in candidate_tokens_set:
                if tokens_tup[:req_prefix_len] != candidate_tokens[:req_prefix_len]:
                    continue
                entry = self._entries.get((model, candidate_tokens))
                if entry is None:
                    continue
                mlcp = self._lcp(entry.model_tokens, req_model)
                if mlcp <= 0:
                    continue
                full_cover = mlcp >= len(entry.model_tokens)
                better = False
                if full_cover and not best_at_level_full_cover:
                    better = True
                elif full_cover == best_at_level_full_cover and mlcp > best_at_level_model_lcp:
                    better = True
                if better:
                    best_at_level_full_cover = full_cover
                    best_at_level_model_lcp = mlcp
                    best_at_level = candidate_tokens
                    best_at_level_entry_model = entry.model_tokens
            if best_at_level is not None:
                best_canon_prefix_len = req_prefix_len
                best_cached_tokens = best_at_level
                best_entry_model_tokens = best_at_level_entry_model
                best_model_lcp = best_at_level_model_lcp
                break

        if best_cached_tokens is None or best_model_lcp <= 0:
            return None, list(req_model), tokens_tup, "miss", 0

        # Extend canonical match inside the current block (kept for accurate
        # canonical_matched_prefix_len reporting — purely telemetry).
        while best_canon_prefix_len < min(len(tokens_tup), len(best_cached_tokens)):
            if tokens_tup[best_canon_prefix_len] == best_cached_tokens[best_canon_prefix_len]:
                best_canon_prefix_len += 1
            else:
                break

        entry = self._extract(model, best_cached_tokens)

        # Prefer trimming to model-space LCP.  Cap at len(req_model) to avoid
        # keeping KV past the new request's length (the caller would start
        # generation from an offset past the input, corrupting state).
        target_model_len = min(best_model_lcp, len(req_model))
        if target_model_len <= 0:
            return None, list(req_model), tokens_tup, "miss", 0

        ok, effective_kv = _trim_to_model_depth(
            entry.prompt_cache, best_entry_model_tokens, target_model_len
        )
        if not ok:
            return None, list(req_model), tokens_tup, "miss", 0

        if effective_kv == len(req_model):
            # Full model-space cover: trim one more token so the model has
            # something to decode from to start generation.
            # P6/1c: `matched_prefix_len` is reported in CANONICAL space for
            # telemetry + M3 threshold comparison.  The trim-by-1 below is
            # in MODEL space and is an implementation detail of the decode
            # bootstrap — do NOT subtract 1 from `best_canon_prefix_len`.
            # Pre-P6 the trimmable branch subtracted 1 (off by canon/model
            # delta for VLM) while the non-trimmable branch did not; the
            # two paths now agree.
            if (
                len(req_model) > 1
                and can_trim_prompt_cache(entry.prompt_cache)
            ):
                trim_prompt_cache(entry.prompt_cache, 1)
                return (
                    entry.prompt_cache,
                    list(req_model)[-1:],
                    best_cached_tokens,
                    "exact",
                    best_canon_prefix_len,
                )
            return (
                entry.prompt_cache,
                [],
                best_cached_tokens,
                "exact",
                best_canon_prefix_len,
            )

        # Shorter (partial) model-space cover: prefill the tail.
        rest = list(req_model)[effective_kv:]
        return (
            entry.prompt_cache,
            rest,
            best_cached_tokens,
            "shorter",
            best_canon_prefix_len,
        )

    def insert_cache(self, model, tokens, prompt_cache, model_tokens=None):
        self.prune_expired()
        tokens_tup = tuple(tokens)
        # Default model_tokens to the canonical cache key for the LM path,
        # where canonical == model.  VLM paths MUST pass model_tokens so the
        # stored KV state can be correctly matched and trimmed on retrieve.
        model_tokens_tup = (
            tuple(model_tokens) if model_tokens is not None else tokens_tup
        )
        key = (model, tokens_tup)
        now = time.time()

        if key in self._entries:
            existing = self._entries[key]
            existing.count += 1
            existing.touched_at = now
            # Refresh model_tokens in case a later call has more accurate
            # information (shouldn't happen in normal flow, but defensive).
            existing.model_tokens = model_tokens_tup
            return

        # 1. Subsumption: Cull redundant prefixes to organically free up space.
        # P6/1d: pass model_tokens so the cull respects model-space distinctness,
        # not just canonical-prefix relation (defense-in-depth against any
        # future normalisation pass that would cross-alias distinct streams).
        self._cull_redundant_prefixes(model, tokens_tup, model_tokens_tup)

        # 2. Insert into flat dictionary
        self._entries[key] = self.CacheEntry(
            prompt_cache, tokens_tup, model_tokens_tup, 1, now
        )

        # 3. Map block hashes
        chain_pairs = _block_chain_hashes(tokens_tup, self.block_size)
        for chain_hash, _ in chain_pairs:
            idx_key = (model, chain_hash)
            if idx_key not in self._block_index:
                self._block_index[idx_key] = set()
            self._block_index[idx_key].add(tokens_tup)

        # 4. Enforce max size using the Cost-Aware Eviction
        while len(self._entries) > self.max_size:
            self._evict_optimal()

    def contains_tokens(self, model, tokens):
        key = (model, tuple(tokens))
        return key in self._entries

    def extract_exact_cache(self, model, tokens):
        if not self.contains_tokens(model, tokens):
            return None
        return self._extract(model, tuple(tokens))

    def delete(self, model, tokens) -> bool:
        """
        Public wrapper around _delete for callers that need to evict a specific
        entry (e.g. the VLM partial-image retreat path: the entry that produced
        the bad cut must be removed so it does not re-win candidate selection
        on the next request — otherwise the retreat is a starvation loop that
        re-prefills the full context forever).

        Returns True if an entry was actually deleted, False if no such key
        existed (safe no-op — callers can invoke unconditionally).
        """
        key = (model, tuple(tokens))
        if key not in self._entries:
            return False
        self._delete(model, tokens)
        return True


PROMPT_CACHE = LRUPromptCache(
    max_size=SETTINGS.prompt_cache_max_entries_global,
    ttl_seconds=SETTINGS.prompt_cache_ttl_seconds,
)


@dataclass(frozen=True)
class SessionContext:
    session_id: str
    parent_session_id: Optional[str]
    branch_id: Optional[str]
    source: str


class SessionIndex:
    @dataclass
    class SessionState:
        parent_session_id: Optional[str]
        touched_at: float
        keys: deque
        anchors: deque

    def __init__(self, max_entries_per_session: int, max_idle_seconds: int):
        self._max_entries_per_session = max(1, max_entries_per_session)
        self._max_idle_seconds = max_idle_seconds
        self._max_anchor_entries = max(4, self._max_entries_per_session * 4)
        self._anchor_stride_tokens = 2048
        self._sessions: Dict[str, SessionIndex.SessionState] = {}

    def _prune_idle(self) -> None:
        if self._max_idle_seconds <= 0:
            return
        now = time.time()
        stale = [
            session_id
            for session_id, state in self._sessions.items()
            if (now - state.touched_at) > self._max_idle_seconds
        ]
        for session_id in stale:
            self._sessions.pop(session_id, None)

    def _lineage_chain(self, session_id: str, max_depth: int = 8) -> List[str]:
        chain: List[str] = []
        seen = set()
        current = session_id
        depth = 0
        while current and current not in seen and depth < max_depth:
            chain.append(current)
            seen.add(current)
            state = self._sessions.get(current)
            if state is None:
                break
            current = state.parent_session_id
            depth += 1
        return chain

    @staticmethod
    def _lcp_len(a: List[int], b: Tuple[int, ...]) -> int:
        limit = min(len(a), len(b))
        idx = 0
        while idx < limit and a[idx] == b[idx]:
            idx += 1
        return idx

    @staticmethod
    def _append_unique_bounded(
        queue: deque, key_tuple: Tuple[int, ...], limit: int
    ) -> None:
        try:
            queue.remove(key_tuple)
        except ValueError:
            pass
        queue.append(key_tuple)
        while len(queue) > limit:
            queue.popleft()

    def register_cache_key(
        self, session_ctx: SessionContext, cache_key: List[int]
    ) -> None:
        self._prune_idle()
        session_id = (session_ctx.session_id or "").strip()
        if not session_id:
            return

        state = self._sessions.get(session_id)
        if state is None:
            state = self.SessionState(
                parent_session_id=session_ctx.parent_session_id,
                touched_at=time.time(),
                keys=deque(),
                anchors=deque(),
            )
            self._sessions[session_id] = state

        if (
            session_ctx.parent_session_id
            and session_ctx.parent_session_id != session_id
        ):
            state.parent_session_id = session_ctx.parent_session_id
        state.touched_at = time.time()

        key_tuple = tuple(cache_key)
        self._append_unique_bounded(
            state.keys, key_tuple, self._max_entries_per_session
        )

        # Keep additional anchor prefixes so branch returns can reuse older stable
        # points even when recent keys are from another branch/tool-heavy turn.
        should_add_anchor = False
        if not state.anchors:
            should_add_anchor = True
        else:
            last_anchor_len = len(state.anchors[-1])
            if (len(key_tuple) - last_anchor_len) >= self._anchor_stride_tokens:
                should_add_anchor = True
        if should_add_anchor:
            self._append_unique_bounded(
                state.anchors, key_tuple, self._max_anchor_entries
            )

    def _selection_from_exact_entry(
        self,
        prompt_tokens: List[int],
        cache_tokens: Tuple[int, ...],
        cache_entry: Any,
    ):
        prefix_len = self._lcp_len(prompt_tokens, cache_tokens)
        if prefix_len <= 0:
            return None

        if len(cache_tokens) > prefix_len:
            if not can_trim_prompt_cache(cache_entry.prompt_cache):
                return None
            trim_prompt_cache(cache_entry.prompt_cache, len(cache_tokens) - prefix_len)

        if prefix_len == len(prompt_tokens):
            if len(prompt_tokens) > 1 and can_trim_prompt_cache(
                cache_entry.prompt_cache
            ):
                trim_prompt_cache(cache_entry.prompt_cache, 1)
                return (
                    cache_entry.prompt_cache,
                    prompt_tokens[-1:],
                    list(cache_tokens),
                    "exact",
                    len(prompt_tokens) - 1,
                )
            return (
                cache_entry.prompt_cache,
                prompt_tokens,
                list(cache_tokens),
                "exact",
                len(prompt_tokens),
            )

        return (
            cache_entry.prompt_cache,
            prompt_tokens[prefix_len:],
            list(cache_tokens),
            "shorter",
            prefix_len,
        )

    def select_best_cache(
        self,
        model_name: str,
        prompt_tokens: List[int],
        session_ctx: SessionContext,
        prompt_cache_store: LRUPromptCache,
        model_tokens: Optional[List[int]] = None,
    ):
        """Always use global cache; session/conversation/thread ID is ignored for lookup."""
        prompt_cache_store.prune_expired()
        self._prune_idle()
        selected = prompt_cache_store.fetch_nearest_cache(
            model_name, prompt_tokens, request_model_tokens=model_tokens
        )
        return (*selected, "global")


SESSION_INDEX = SessionIndex(
    max_entries_per_session=SETTINGS.prompt_cache_max_entries_per_session,
    max_idle_seconds=SETTINGS.prompt_cache_session_max_idle_seconds,
)


# --- MESSAGE-AWARE STABLE-PREFIX CACHE (Phase 2) ---


@dataclass
class _SessionTurnRecord:
    """
    Lightweight per-session record of the last completed turn's message structure.

    `canonical_messages` is the `_canonicalize_messages` output — the SAME form
    the cache-key pipeline uses.  Storing this (rather than the original raw
    messages) is what lets the write-guard and `_message_diff` treat OpenClaw's
    per-turn drift (message_id, subagent Stats runtime, Inbound Context block
    variants, timestamps, system-reminder injections) as equal — because they
    ARE equal as far as the cache key is concerned.  Pre-G3 the guard compared
    whitespace-stripped originals, any one drifted field froze the store and
    the M3 stable-prefix layer stopped advancing for the rest of the session.
    """

    canonical_messages: List[Dict[str, Any]]  # canonicalized form — for diff/guard only
    msg_token_lens: List[int]  # token count for each message (in order)
    total_prompt_tokens: (
        int  # sum(msg_token_lens); equals len(prompt_tokens) for that turn
    )
    touched_at: float


# Global store: session_id -> _SessionTurnRecord
# Protected by prompt_cache_lock (same lock used for PROMPT_CACHE).
SESSION_TURN_STORE: Dict[str, _SessionTurnRecord] = {}
_SESSION_TURN_MAX_IDLE_SECONDS: int = SETTINGS.prompt_cache_session_max_idle_seconds


def _normalize_message_content_for_diff(msg: Dict[str, Any]) -> str:
    """
    Return a normalised string representation of a message's content for use
    ONLY as a diff key. The original message is never modified.

    Currently strips:
    - Leading/trailing whitespace differences (trailing space is a real FP-1 trigger)
    - No other normalisation until confirmed from real OpenCode/OpenClaw logs.

    M4 IMPLEMENTATION NOTE: When real session logs from OpenCode/OpenClaw are
    audited, add confirmed volatile-field stripping here (timestamps, system-reminder
    injections, etc.). The post-render _scrub_cache_key() patterns are a reference
    but are applied at the serialised-string level — they need to be adapted here at
    the per-message level after real-traffic confirmation.
    """
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Multi-part content (e.g. VLM image+text). Use JSON with stripped text parts.
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append({"type": "text", "text": part.get("text", "").strip()})
                else:
                    parts.append(part)
            else:
                parts.append(part)
        return json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return json.dumps(content, sort_keys=True, ensure_ascii=False)


def _message_diff(
    prev_msgs: List[Dict[str, Any]],
    curr_msgs: List[Dict[str, Any]],
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Diff two message lists at message-boundary level.

    Returns:
        stable_prefix_count (int): number of messages from the START of both lists
            that are normalisation-identical in role and content. These messages'
            KV state can be safely reused.
        descriptors (list): list of change descriptors for telemetry/debugging.

    Rules:
    - Two messages are "equal" if their role is identical AND their normalised
      content (_normalize_message_content_for_diff) is identical.
    - Only the LEADING equal block contributes to stable_prefix_count.
      If messages [0..K-1] are equal and message[K] differs, stable_prefix_count = K.
    - Later equal blocks (after a change) are NOT counted — RoPE position
      correctness requires contiguous prefix reuse only.
    """
    stable_prefix_count = 0
    for i, (prev, curr) in enumerate(zip(prev_msgs, curr_msgs)):
        if prev.get("role") == curr.get("role") and _normalize_message_content_for_diff(
            prev
        ) == _normalize_message_content_for_diff(curr):
            stable_prefix_count = i + 1
        else:
            break

    descriptors: List[Dict[str, Any]] = []
    n = max(len(prev_msgs), len(curr_msgs))
    for i in range(n):
        p = prev_msgs[i] if i < len(prev_msgs) else None
        c = curr_msgs[i] if i < len(curr_msgs) else None
        if p is None:
            descriptors.append({"idx": i, "op": "insert", "role": c.get("role")})
        elif c is None:
            descriptors.append({"idx": i, "op": "delete", "role": p.get("role")})
        elif p.get("role") != c.get("role") or _normalize_message_content_for_diff(
            p
        ) != _normalize_message_content_for_diff(c):
            descriptors.append(
                {
                    "idx": i,
                    "op": "replace",
                    "role_prev": p.get("role"),
                    "role_curr": c.get("role"),
                }
            )

    return stable_prefix_count, descriptors


def _stable_prefix_token_len(
    session_id: str,
    curr_msgs: List[Dict[str, Any]],
) -> Tuple[int, int, List[Dict[str, Any]]]:
    """
    For a given session and incoming message list, determine the stable prefix
    in tokens by consulting the SESSION_TURN_STORE.

    Returns:
        stable_token_len (int): number of tokens at the start of the prompt that
            are known-stable from the previous turn. 0 if no prior turn or no match.
        stable_msg_count (int): number of leading messages that are stable.
        descriptors (list): diff descriptors for telemetry.

    Must be called while holding prompt_cache_lock (reads SESSION_TURN_STORE).
    """
    record = SESSION_TURN_STORE.get(session_id)
    if record is None:
        return 0, 0, []

    # G3: diff in canonical form.  The stored record already holds canonical
    # messages; the incoming list must be canonicalised before comparison so
    # that OpenClaw-style per-turn drift (message_id, Stats runtime, Inbound
    # Context variants) collapses to the same canonical sentinels and the
    # leading block is correctly detected as stable.
    _, canonical_curr = _canonicalize_messages(curr_msgs)
    stable_msg_count, descriptors = _message_diff(
        record.canonical_messages, canonical_curr
    )
    if stable_msg_count == 0:
        return 0, 0, descriptors

    # Sum the token lengths of the stable prefix messages
    stable_token_len = sum(record.msg_token_lens[:stable_msg_count])
    return stable_token_len, stable_msg_count, descriptors


def _compute_msg_token_boundaries(
    messages: List[Dict[str, Any]],
    prompt_tokens: List[int],
) -> List[int]:
    """
    Compute accurate per-message token boundary lengths using cumulative chat-template
    rendering. Each boundary is the cumulative token count up to and including that
    message in the rendered prompt.

    Strategy: render messages[0..i] (without generation prompt) through the actual
    chat template and measure the token count. The per-message length is the diff
    between consecutive cumulative counts.

    Returns a list of per-message token counts (same length as messages).
    The last message gets len(prompt_tokens) - sum(prev) as ground truth.

    For VLM prompts: tokenizer may not have apply_chat_template; falls back to
    equal distribution with last-message remainder correction.
    """
    n = len(messages)
    if n == 0:
        return []

    msg_token_lens: List[int] = []

    # Prefer apply_chat_template if available (gives exact boundaries)
    if (
        tokenizer is not None
        and hasattr(tokenizer, "apply_chat_template")
        and getattr(tokenizer, "chat_template", None) is not None
        and not is_vlm
    ):
        try:
            cumulative_lengths: List[int] = []
            prev_len = 0
            for i in range(n):
                # Render prefix messages[0..i] without generation prompt so the
                # boundary lands exactly at the end of message i.
                # We suppress add_generation_prompt to avoid including the assistant
                # start token in the boundary count.
                prefix_toks = tokenizer.apply_chat_template(
                    messages[: i + 1],
                    tokenize=True,
                    add_generation_prompt=False,
                    **(
                        {"preserve_thinking": True}
                        if SETTINGS.preserve_thinking
                        and _PRESERVE_THINKING_SUPPORTED
                        else {}
                    ),
                )
                if isinstance(prefix_toks, list):
                    cur_len = len(prefix_toks)
                else:
                    cur_len = prev_len
                msg_token_lens.append(max(0, cur_len - prev_len))
                prev_len = cur_len
            # Correct the last bucket using prev_len, which after the loop holds the
            # token count for the full message list rendered WITHOUT a generation
            # prompt (last loop iteration called apply_chat_template(messages[:n], ...,
            # add_generation_prompt=False)). We intentionally do NOT use
            # len(prompt_tokens) here because prompt_tokens was rendered with
            # add_generation_prompt=True (and enable_thinking=True for Qwen3), which
            # appends tokens such as "<|im_start|>assistant\n<think>\n\n". If those
            # tokens were absorbed into the last-message bucket,
            # _stable_prefix_token_len would sum past the real message content into
            # the generation-prompt region. On the next request the same position
            # holds a different token depending on how the new response starts
            # (e.g. <think>\n ID 198 vs <think>\n\n ID 271), causing the
            # "cache divergence at <think>" bug with Qwen3 + tool calls.
            # prev_len gives a boundary that stops exactly at the last real message.
            if msg_token_lens:
                prefix_sum = sum(msg_token_lens[:-1])
                msg_token_lens[-1] = max(0, prev_len - prefix_sum)
            return msg_token_lens
        except Exception:
            pass  # Fall through to approximation

    # Fallback: divide evenly, correct last bucket with remainder
    per = len(prompt_tokens) // max(1, n)
    msg_token_lens = [per] * (n - 1) + [max(0, len(prompt_tokens) - per * (n - 1))]
    return msg_token_lens


def _update_session_turn_store(
    session_id: str,
    messages: List[Dict[str, Any]],
    prompt_tokens: List[int],
) -> None:
    """
    Record the per-message token boundary information for this session's completed turn.
    Must be called while holding prompt_cache_lock.

    Per-message token lengths are computed via cumulative chat-template rendering
    (_compute_msg_token_boundaries) to give exact boundaries. The stable-prefix
    secondary lookup depends on these boundaries being accurate: if the boundary
    is over-estimated, the lookup prefix extends into the next (changed) message
    and the block-hash mismatches.
    """
    if not session_id or not messages:
        return

    # Prune stale entries. A value of 0 means "never expire" (infinite TTL),
    # consistent with LRUPromptCache._is_expired which returns False when
    # ttl_seconds <= 0. Without this guard, _SESSION_TURN_MAX_IDLE_SECONDS=0
    # makes (now - touched_at) > 0 always True, pruning every record on every
    # write and silently destroying the stable-prefix mechanism for all sessions.
    now = time.time()
    if _SESSION_TURN_MAX_IDLE_SECONDS > 0:
        stale = [
            sid
            for sid, rec in SESSION_TURN_STORE.items()
            if (now - rec.touched_at) > _SESSION_TURN_MAX_IDLE_SECONDS
        ]
        for sid in stale:
            del SESSION_TURN_STORE[sid]

    # G3: canonicalise the incoming messages BEFORE the write guard runs so
    # that OpenClaw's per-turn drift on message_id / Stats runtime / Inbound
    # Context block / timestamps / system-reminder injections is treated as
    # equal — because it IS equal in canonical (cache-key) space.  Pre-G3
    # the guard used whitespace-only normalisation, so a single drifted
    # field froze the store for the remainder of the session and the M3
    # stable-prefix layer stopped advancing.
    _, canonical_curr = _canonicalize_messages(messages)

    # Guard: only write if the new canonical message list is a strict append
    # of the existing record's canonical form.  If a prior canonical message
    # differs, this request is from a sub-agent or parallel branch with a
    # structurally different history on the same session_id — clobbering the
    # orchestrator's record would corrupt the next orchestrator turn's diff.
    existing = SESSION_TURN_STORE.get(session_id)
    if existing is not None:
        prev = existing.canonical_messages
        n_prev = len(prev)
        if len(canonical_curr) < n_prev:
            # Shorter than what we already have — definitely not an append. Skip.
            return
        for i in range(n_prev):
            if prev[i].get("role") != canonical_curr[i].get(
                "role"
            ) or _normalize_message_content_for_diff(
                prev[i]
            ) != _normalize_message_content_for_diff(canonical_curr[i]):
                # Canonical-form divergence — a prior message really changed
                # semantically.  Skip to preserve the best-known linear record.
                return

    # msg_token_lens is derived from the ORIGINAL (template-prepared) messages
    # so per-message boundaries correspond to the render the model actually
    # sees.  The pre-existing original-vs-canonical token-count mismatch when
    # canonicalisation collapses volatile spans is orthogonal to G3 and tracked
    # as a P6 cleanup item.
    msg_token_lens = _compute_msg_token_boundaries(messages, prompt_tokens)

    SESSION_TURN_STORE[session_id] = _SessionTurnRecord(
        canonical_messages=canonical_curr,
        msg_token_lens=msg_token_lens,
        total_prompt_tokens=len(prompt_tokens),
        touched_at=now,
    )


class CacheSessionTranscriptLogger:
    def __init__(self, cache_session_id: str):
        now = datetime.now()
        day_dir = SETTINGS.log_root / now.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        self.path = day_dir / f"cache-session-{cache_session_id}.log"
        if not self.path.exists():
            self._write_line(f"# cache_session_id={cache_session_id}")
            self._write_line(f"# created_at={self._ts()}")

    def _ts(self):
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

    def _write_line(self, line):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")

    def log(self, direction, payload, request_id: str):
        self._write_line(
            f"[{self._ts()}] request_id={request_id} direction={direction}"
        )
        if isinstance(payload, (dict, list)):
            self._write_line(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            self._write_line(str(payload))
        self._write_line("")


def _cache_session_id(tokens: List[int]) -> str:
    raw = ",".join(str(tok) for tok in tokens[:1024])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_log_session_id(
    session_ctx: SessionContext, cache_session_tokens: List[int]
) -> str:
    # Prefer stable conversation/session identifier for grouped diagnostics.
    session_raw = (session_ctx.session_id or "").strip()
    if session_raw:
        return hashlib.sha1(session_raw.encode("utf-8")).hexdigest()[:16]
    return _cache_session_id(cache_session_tokens)


def _flatten_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_content = ""
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_content += part.get("text", "")
            elif isinstance(part, str):
                text_content += part
        return text_content
    return content


def _prepare_messages_for_template(messages):
    normalized = []
    for msg in messages:
        m = dict(msg)
        m["content"] = _flatten_content(m.get("content", ""))

        # OpenAI tool_calls use function.arguments as a JSON string.
        # GLM's template expects arguments to be a mapping so it can iterate items().
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            fixed_tool_calls = []
            for tc in m["tool_calls"]:
                tc_copy = dict(tc)
                fn = tc_copy.get("function")
                if isinstance(fn, dict):
                    fn_copy = dict(fn)
                    fn_name = str(fn_copy.get("name", "")).strip().lower()
                    args = fn_copy.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn_copy["arguments"] = json.loads(args)
                        except Exception:
                            fn_copy["arguments"] = {"raw": args}
                    if (
                        SETTINGS.normalize_write_tool_content_for_prompt
                        and fn_name == "write"
                        and isinstance(fn_copy.get("arguments"), dict)
                        and "content" in fn_copy["arguments"]
                    ):
                        # Keep write semantics (path + write intent) while stabilizing
                        # prompt tokens by replacing huge inline content with a digest tag.
                        raw_content = fn_copy["arguments"]["content"]
                        if isinstance(raw_content, str):
                            raw_bytes = raw_content.encode("utf-8")
                            digest = hashlib.sha1(raw_bytes).hexdigest()[:16]
                            placeholder = (
                                f"__WRITE_CONTENT_OMITTED__sha1={digest};"
                                f"bytes={len(raw_bytes)};chars={len(raw_content)};"
                                f"lines={raw_content.count(chr(10)) + 1}__"
                            )
                        else:
                            serialized = json.dumps(
                                raw_content, ensure_ascii=False, sort_keys=True
                            )
                            raw_bytes = serialized.encode("utf-8")
                            digest = hashlib.sha1(raw_bytes).hexdigest()[:16]
                            placeholder = (
                                f"__WRITE_CONTENT_OMITTED_NONSTRING__sha1={digest};"
                                f"bytes={len(raw_bytes)}__"
                            )
                        args_copy = dict(fn_copy["arguments"])
                        args_copy["content"] = placeholder
                        fn_copy["arguments"] = args_copy
                    tc_copy["function"] = fn_copy
                fixed_tool_calls.append(tc_copy)
            m["tool_calls"] = fixed_tool_calls

        normalized.append(m)
    return normalized


def _resolve_vlm_image_token_ids(config: Any) -> Tuple[Optional[int], Optional[int]]:
    """
    Resolve the (image_token_id, video_token_id) that mlx-vlm's VLM forward
    uses to match image_pad positions against pixel_values features.

    Different mlx-vlm model configs use one of several key names:
      image_token_id, image_token_index, image_pad_token_id,
      mm_image_token_id
    Returns (None, None) if neither is resolvable — caller must fall back to
    a length-based heuristic.
    """
    if not isinstance(config, dict):
        # Some VLM config objects are attribute-only; coerce if possible.
        try:
            config = dict(vars(config))
        except Exception:
            return None, None

    def _first_int(keys: Tuple[str, ...]) -> Optional[int]:
        for k in keys:
            if k in config and isinstance(config[k], (int, float)):
                try:
                    return int(config[k])
                except (TypeError, ValueError):
                    continue
        # Fall back into nested `text_config` / `vision_config` (Qwen2_VL etc.)
        for nested in ("text_config", "vision_config", "config"):
            sub = config.get(nested)
            if isinstance(sub, dict):
                for k in keys:
                    if k in sub and isinstance(sub[k], (int, float)):
                        try:
                            return int(sub[k])
                        except (TypeError, ValueError):
                            continue
        return None

    image_id = _first_int((
        "image_token_id",
        "image_token_index",
        "image_pad_token_id",
        "mm_image_token_id",
    ))
    video_id = _first_int((
        "video_token_id",
        "video_token_index",
        "video_pad_token_id",
    ))
    return image_id, video_id


# Resolved once at startup (after model load) — may be None for text-only LM.
VLM_IMAGE_TOKEN_ID: Optional[int] = None
VLM_VIDEO_TOKEN_ID: Optional[int] = None


# --- G1: image-identity cache-key correctness ---
# The canonical cache key for VLM encodes each image placeholder as a single
# `<|image_pad|>` token (the text tokeniser does not expand to per-feature
# tokens — only mlx-vlm's prepare_inputs does, on the model path).  Two
# requests that share message structure but carry DIFFERENT images therefore
# produce IDENTICAL canonical token streams.  The block-hash index and
# model-space LCP then accept image-A's cached KV for image-B's request and
# silently serve wrong answers (see .context/LESSONS.md and GOALS.md G1).
#
# Fix: inject a deterministic 2-token marker per image into the canonical
# token stream right after each `<|image_pad|>` position, in message order.
# Markers live in a reserved id range (`_IMG_MARKER_BASE`..`_IMG_MARKER_MASK`)
# above any real tokenizer vocab and safely inside uint32 for the block-hash
# chain.  They ONLY appear in canonical cache-key tokens; `model_tokens` and
# the model's view are untouched — the dual-pipeline invariant is preserved.
_IMG_MARKER_BASE: int = 0x7F00_0000  # 2_130_706_432 — above any real vocab
_IMG_MARKER_MASK: int = 0x00FF_FFFF  # 24 bits of hash per synthetic token


def _image_identity_markers(images: List[Any]) -> List[List[int]]:
    """
    Compute a stable 2-token marker per image for the canonical cache key.

    Hashing strategy by source type:
    - PIL Image: SHA-256 of raw pixel bytes + size tuple.
    - data URL ("data:image/..;base64,..."): SHA-256 of the decoded payload.
    - bare URL string (http/file/s3/etc.): SHA-256 of the URL string itself
      (the best we can do without fetching; collisions imply identical URLs
      which are intended to match).

    Each returned marker is two uint32 tokens in the reserved range; combined
    they carry 48 bits of entropy (collision probability ≈ 3.6e-15 between
    any two images, which is orders of magnitude below the rest of the
    correctness envelope).

    Never raises — on any parse/hash failure the marker degrades to a stable
    placeholder so the cache key stays deterministic instead of diverging.
    """
    import base64

    out: List[List[int]] = []
    for src in images:
        try:
            if hasattr(src, "tobytes") and hasattr(src, "size"):
                raw = src.tobytes() + repr(getattr(src, "size", ())).encode("utf-8")
                digest = hashlib.sha256(raw).digest()
            elif isinstance(src, str):
                if src.startswith("data:") and "," in src:
                    _, b64 = src.split(",", 1)
                    try:
                        raw = base64.b64decode(b64)
                    except Exception:
                        raw = src.encode("utf-8")
                    digest = hashlib.sha256(raw).digest()
                else:
                    digest = hashlib.sha256(src.encode("utf-8")).digest()
            else:
                digest = hashlib.sha256(repr(src).encode("utf-8")).digest()
        except Exception:
            digest = hashlib.sha256(b"__img_hash_failed__").digest()
        d = int.from_bytes(digest[:6], "big")  # 48 bits
        t0 = _IMG_MARKER_BASE | (d & _IMG_MARKER_MASK)
        t1 = _IMG_MARKER_BASE | ((d >> 24) & _IMG_MARKER_MASK)
        out.append([t0, t1])
    return out


def _inject_image_markers(
    canon_ids: List[int],
    markers: List[List[int]],
    pad_id: int,
) -> List[int]:
    """
    Append each marker to `canon_ids` immediately after the contiguous run of
    `pad_id` tokens that corresponds to the i-th image, in order.

    Text-only tokenisation normally collapses each `<|image_pad|>` placeholder
    into a single token, but contiguous runs are tolerated for robustness
    against template variants.  If pad runs and markers have different counts
    we inject up to `min(len)` and leave the rest untouched — degraded but
    still deterministic (consistency across turns is what matters for cache
    lookup).
    """
    if not markers or not canon_ids:
        return list(canon_ids)
    out: List[int] = []
    n = len(canon_ids)
    i = 0
    m_idx = 0
    while i < n:
        tok = canon_ids[i]
        if tok == pad_id and m_idx < len(markers):
            # Absorb the full contiguous pad run, then emit the marker.
            j = i
            while j < n and canon_ids[j] == pad_id:
                out.append(canon_ids[j])
                j += 1
            out.extend(markers[m_idx])
            m_idx += 1
            i = j
            continue
        out.append(tok)
        i += 1
    return out


def _messages_have_images(messages: List[Dict[str, Any]]) -> bool:
    """True if any message has content list containing image_url or input_image."""
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                t = (part.get("type") or "").strip().lower()
                if t in ("image_url", "input_image"):
                    return True
                if "image_url" in part or "input_image" in part:
                    return True
    return False


def _extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    """
    Extract image sources from OpenAI-style content (type image_url / input_image).
    Returns list in message order (data URL strings or PIL Images for prepare_inputs).
    """
    import base64
    from io import BytesIO

    images = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            url = None
            if part.get("type") == "image_url":
                u = part.get("image_url") or {}
                url = u.get("url") if isinstance(u, dict) else None
            elif part.get("type") == "input_image":
                u = part.get("input_image") or part.get("image_url") or {}
                url = (
                    u.get("url")
                    if isinstance(u, dict)
                    else u
                    if isinstance(u, str)
                    else None
                )
            if not url or not isinstance(url, str):
                continue
            url = url.strip()
            if url.startswith("data:image/") and "," in url:
                try:
                    _, b64 = url.split(",", 1)
                    from PIL import Image

                    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
                    images.append(img)
                except Exception:
                    images.append(url)
            else:
                images.append(url)
    return images


def _prepare_messages_for_vlm(
    messages: List[Dict[str, Any]], tools: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Normalize messages for VLM: fix tool_calls like _prepare_messages_for_template,
    but preserve content as list (text + image_url) so get_chat_template can insert image tokens.
    """
    normalized = []
    for msg in messages:
        m = dict(msg)
        content = m.get("content", "")
        if isinstance(content, list):
            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text") or part.get("content") or ""
                    # NOTE: do NOT normalize here — model must see original content.
                    # Canonicalization for cache key happens via _canonicalize_messages()
                    # before rendering, and only in the cache key pipeline.
                    new_content.append({**part, "text": text})
                else:
                    new_content.append(part)
            m["content"] = new_content
        elif isinstance(content, str):
            m["content"] = content
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            fixed_tool_calls = []
            for tc in m["tool_calls"]:
                tc_copy = dict(tc)
                fn = tc_copy.get("function")
                if isinstance(fn, dict):
                    fn_copy = dict(fn)
                    args = fn_copy.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn_copy["arguments"] = json.loads(args)
                        except Exception:
                            fn_copy["arguments"] = {"raw": args}
                    tc_copy["function"] = fn_copy
                fixed_tool_calls.append(tc_copy)
            m["tool_calls"] = fixed_tool_calls
        normalized.append(m)
    return normalized


def _vlm_prompt_and_inputs(
    processor_any,
    config: Dict[str, Any],
    messages: List[Dict[str, Any]],
    images: List[Any],
    tools: Optional[Any] = None,
    enable_thinking: Optional[bool] = None,
    **kwargs: Any,
) -> Tuple[Any, Any, Any, Any]:
    """
    Build formatted prompt and run prepare_inputs for VLM using native chat templates.
    Returns (input_ids, pixel_values, mask, vlm_kwargs) where input_ids is mx.array; mask may be None.
    """
    template_kwargs = dict(kwargs)
    if tools is not None:
        template_kwargs["tools"] = tools
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    template_processor = None
    if processor_any is not None and hasattr(processor_any, "apply_chat_template"):
        if getattr(processor_any, "chat_template", None) is not None:
            template_processor = processor_any
    if (
        template_processor is None
        and getattr(processor_any, "tokenizer", None) is not None
    ):
        tok = processor_any.tokenizer
        if (
            hasattr(tok, "apply_chat_template")
            and getattr(tok, "chat_template", None) is not None
        ):
            template_processor = tok

    formatted = ""
    if template_processor is not None:
        try:
            formatted = template_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            if isinstance(formatted, list):
                formatted = formatted[0] if formatted else ""
            formatted = str(formatted)
        except Exception:
            formatted = ""

    # Fallback to mlx_vlm's get_chat_template if HF template fails/is missing
    if not formatted:
        out = get_chat_template(
            processor_any,
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
        )
        if isinstance(out, list):
            out = out[0].get("content", "") if out else ""
        formatted = str(out) if out else ""

    # NOTE: do NOT scrub the formatted string here — this is the model input.
    # Post-render scrubbing for the cache key is applied in do_POST on the
    # cache_prompt_raw string (canonical pipeline), never on the model input.

    inputs = vlm_prepare_inputs(
        processor_any,
        images=images if images else None,
        prompts=formatted,
        add_special_tokens=True,
        return_tensors="mlx",
    )

    input_ids = inputs.get("input_ids")
    pixel_values = inputs.get("pixel_values")
    mask = inputs.get("attention_mask")
    vlm_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    return input_ids, pixel_values, mask, vlm_kwargs


def _vlm_sync_before_generation(pixel_values: Any, mask: Any) -> None:
    """
    Flush Metal work before VLM generation to prevent PyTorch and MLX from colliding.
    """
    try:
        import torch

        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()  # CRITICAL: Force PyTorch to release all Metal encoders
    except Exception:
        pass

    to_eval = []
    if pixel_values is not None and hasattr(pixel_values, "shape"):
        to_eval.append(pixel_values)
    if mask is not None and hasattr(mask, "shape"):
        to_eval.append(mask)
    if to_eval:
        try:
            import mlx.core as mx

            mx.eval(*to_eval)
        except Exception:
            pass


def _should_enable_thinking(body):
    if isinstance(body.get("enable_thinking"), bool):
        return body["enable_thinking"]
    # Default to enabled
    return SETTINGS.default_thinking


# P6/1e: resolved once at startup via `_probe_preserve_thinking_support`.
# None = probe hasn't run yet (open to either) — effectively True on first call.
# After probe: concrete True / False.
_PRESERVE_THINKING_SUPPORTED: Optional[bool] = None


def _probe_preserve_thinking_support(targets: List[Any]) -> bool:
    """
    P6/1e: determine whether `preserve_thinking` can be forwarded to the
    active tokenizer/processor's `apply_chat_template` without raising.
    HuggingFace tokenizers generally accept unknown Jinja-context kwargs,
    but `ProcessorMixin.apply_chat_template` signatures vary — a future
    mlx-vlm / transformers upgrade with stricter kwarg checking would
    TypeError every request.  Probing once at startup (after model load)
    and caching the result means zero per-request cost and a defensive
    posture.

    All targets must accept the kwarg for us to return True; if ANY one
    rejects it we disable forwarding everywhere (the render paths for
    cache-key and model-input must stay consistent).

    Probe payload is intentionally minimal (one short user message) so
    template rendering cost is trivial.  Exceptions during the probe
    fall through to False (safer — no forward means today's behaviour
    without the kwarg, which is what the pre-P2 server shipped).
    """
    probe_msgs = [{"role": "user", "content": "probe"}]
    for target in targets:
        if target is None:
            continue
        fn = getattr(target, "apply_chat_template", None)
        chat_tpl = getattr(target, "chat_template", None)
        if fn is None or chat_tpl is None:
            # No template available on this target — can't probe; assume
            # not supported to stay on the safe side for its paths.
            return False
        try:
            fn(probe_msgs, tokenize=False, add_generation_prompt=False,
               preserve_thinking=True)
        except TypeError:
            return False
        except Exception:
            # Non-kwarg error (e.g. template doesn't handle empty context) —
            # assume kwarg itself is fine; errors unrelated to preserve_thinking
            # will surface on real requests anyway.
            pass
    return True


def _chat_template_extras(enable_thinking: Optional[bool]) -> Dict[str, Any]:
    """
    Build kwargs common to every apply_chat_template call so cache-key path and model-input
    path render identically. `preserve_thinking` keeps prior assistant <think>...</think>
    blocks in the history (critical for Qwen3.6). Templates that don't reference the flag
    ignore it; see chat_template.jinja `{% if preserve_thinking is defined ... %}`.

    P6/1e: only forward `preserve_thinking` when the startup probe confirmed
    the active tokenizer/processor accepts the kwarg.  Pre-P6 this was
    forwarded unconditionally and a stricter-kwarg transformers upgrade
    would break every request.
    """
    extras: Dict[str, Any] = {}
    if enable_thinking is not None:
        extras["enable_thinking"] = enable_thinking
    if SETTINGS.preserve_thinking and _PRESERVE_THINKING_SUPPORTED:
        extras["preserve_thinking"] = True
    return extras


def _reasoning_level_to_enable_thinking(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"off", "none", "disable", "disabled", "false", "0", "no"}:
            return False
        if normalized in {
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "default",
            "on",
            "enabled",
            "true",
            "1",
            "yes",
            "auto",
        }:
            return True
    return None


def _extract_enable_thinking(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine enable_thinking from explicit flag first, then common reasoning fields.

    Supported sources:
    - enable_thinking: bool
    - thinking: str|bool|dict (OpenClaw / Anthropic style)
    - reasoning_effort: str (OpenAI / LiteLLM style)
    - reasoning: str|dict (Responses-style)
    - metadata.{thinking, reasoning, reasoning_effort}
    - extra_body.{thinking, reasoning, reasoning_effort}
    """
    if isinstance(body.get("enable_thinking"), bool):
        return {
            "enable_thinking": body["enable_thinking"],
            "source": "enable_thinking",
            "raw": body["enable_thinking"],
        }

    for key in ("thinking", "reasoning_effort"):
        mapped = _reasoning_level_to_enable_thinking(body.get(key))
        if mapped is not None:
            return {
                "enable_thinking": mapped,
                "source": key,
                "raw": body.get(key),
            }

    reasoning_obj = body.get("reasoning")
    if isinstance(reasoning_obj, dict):
        for nested_key in ("enabled", "effort", "level", "type"):
            mapped = _reasoning_level_to_enable_thinking(reasoning_obj.get(nested_key))
            if mapped is not None:
                return {
                    "enable_thinking": mapped,
                    "source": f"reasoning.{nested_key}",
                    "raw": reasoning_obj.get(nested_key),
                }
    else:
        mapped = _reasoning_level_to_enable_thinking(reasoning_obj)
        if mapped is not None:
            return {
                "enable_thinking": mapped,
                "source": "reasoning",
                "raw": reasoning_obj,
            }

    for container_key in ("metadata", "extra_body"):
        container = body.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ("thinking", "reasoning_effort"):
            mapped = _reasoning_level_to_enable_thinking(container.get(key))
            if mapped is not None:
                return {
                    "enable_thinking": mapped,
                    "source": f"{container_key}.{key}",
                    "raw": container.get(key),
                }
        nested_reasoning = container.get("reasoning")
        if isinstance(nested_reasoning, dict):
            for nested_key in ("enabled", "effort", "level", "type"):
                mapped = _reasoning_level_to_enable_thinking(
                    nested_reasoning.get(nested_key)
                )
                if mapped is not None:
                    return {
                        "enable_thinking": mapped,
                        "source": f"{container_key}.reasoning.{nested_key}",
                        "raw": nested_reasoning.get(nested_key),
                    }
        else:
            mapped = _reasoning_level_to_enable_thinking(nested_reasoning)
            if mapped is not None:
                return {
                    "enable_thinking": mapped,
                    "source": f"{container_key}.reasoning",
                    "raw": nested_reasoning,
                }

    return {
        "enable_thinking": _should_enable_thinking(body),
        "source": "default",
        "raw": None,
    }


def _normalize_assistant_text(text, enable_thinking, model_family):
    if not isinstance(text, str):
        return text
    # Qwen3 output can be sensitive to synthetic prefix injection.
    # Keep raw text stable so next-turn prompt tokens stay cache-friendly.
    if model_family == "qwen3":
        return text
    if enable_thinking and text and not text.lstrip().startswith("<think>"):
        return "<think>" + text
    return text


def _coerce_arg_value(raw_value):
    value = raw_value.strip()
    try:
        return json.loads(value)
    except Exception:
        return value


def _extract_openai_tool_calls(text, model_family):
    if not isinstance(text, str) or "<tool_call>" not in text:
        return text, []

    def _parse_legacy_block(body):
        name_match = re.match(r"^([^\s<]+)", body)
        if not name_match:
            return None
        tool_name = name_match.group(1).strip()
        if not tool_name:
            return None
        args = {}
        for arg_key, arg_value in ARG_PAIR_PATTERN.findall(body):
            key = arg_key.strip()
            if not key:
                continue
            args[key] = _coerce_arg_value(arg_value)
        return {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }

    def _parse_qwen_block(body):
        function_match = QWEN_FUNCTION_PATTERN.search(body)
        if not function_match:
            return None
        tool_name = function_match.group(1).strip()
        fn_body = function_match.group(2)
        if not tool_name:
            return None
        args = {}
        for key, value in QWEN_PARAMETER_PATTERN.findall(fn_body):
            param_key = key.strip()
            if not param_key:
                continue
            args[param_key] = _coerce_arg_value(value)
        return {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        }

    tool_calls = []
    remove_spans = []

    for match in TOOL_CALL_PATTERN.finditer(text):
        body = match.group(1).strip()
        if not body:
            continue
        parsed = None
        if model_family == "qwen3":
            parsed = _parse_qwen_block(body)
            if parsed is None:
                parsed = _parse_legacy_block(body)
        else:
            parsed = _parse_legacy_block(body)
            if parsed is None:
                parsed = _parse_qwen_block(body)
        if parsed is None:
            continue
        tool_calls.append(parsed)
        remove_spans.append(match.span())

    if not remove_spans:
        return text, tool_calls

    cleaned_parts = []
    cursor = 0
    for start, end in remove_spans:
        if start > cursor:
            cleaned_parts.append(text[cursor:start])
        cursor = end
    if cursor < len(text):
        cleaned_parts.append(text[cursor:])
    cleaned_text = "".join(cleaned_parts).strip()
    return cleaned_text, tool_calls


def _tokenize_prompt(prompt):
    if isinstance(prompt, str):
        add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token
        )
        return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if isinstance(prompt, list):
        return prompt
    return list(prompt)


def _insert_cache_entries(
    model_name: str,
    session_ctx: "SessionContext",
    cache_key: List[int],
    prompt_cache: Any,
    generated_tokens: List[int],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    prompt_model_tokens: Optional[List[int]] = None,
) -> None:
    """
    Insert the standard full-turn cache entry and, for tool-call or VLM turns,
    also insert a prompt-only checkpoint. The checkpoint helps next-turn prefix
    reuse when provider-side tool-call serialisation differs between turns, or
    when VLM vision tokens change the effective prompt boundary.

    `prompt_model_tokens` is the MODEL-space token sequence (pre-generation)
    that actually built the KV state.  For VLM, this includes image-pad token
    expansion and differs from the canonical `cache_key`.  For LM paths,
    `prompt_model_tokens` defaults to the pre-generation canonical prefix.

    For plain LM turns without tool calls the full-key entry alone is sufficient:
    the next request will get a 'shorter' hit via _cull_redundant_prefixes or
    trim, without requiring a second deepcopy here. Gating the deepcopy prevents
    two full KV tensors (~2–3 GB each at 20k tokens) from living simultaneously
    in GPU memory on every non-tool turn.
    """
    gen_len = len(generated_tokens or [])
    canonical_prompt_len = max(0, len(cache_key) - gen_len)
    if prompt_model_tokens is None:
        prompt_model_tokens_list = list(cache_key[:canonical_prompt_len])
    else:
        prompt_model_tokens_list = list(prompt_model_tokens)
    # Full-turn model tokens = prompt model tokens + generated tokens.
    model_tokens_full = prompt_model_tokens_list + list(generated_tokens or [])

    # Only insert a prompt-only checkpoint for turns where it actually helps:
    # - tool_calls: next turn's prompt differs from cache key due to serialisation
    # - is_vlm: VLM requests need an explicit prefix entry for vision-token reuse
    # Plain LM turns get 'shorter' cache hits from the full-key entry alone.
    if (
        (tool_calls or is_vlm)
        and generated_tokens
        and can_trim_prompt_cache(prompt_cache)
        and len(cache_key) > len(generated_tokens)
    ):
        try:
            prompt_only_cache = copy.deepcopy(prompt_cache)
            trim_prompt_cache(prompt_only_cache, len(generated_tokens))
            prompt_only_key = cache_key[: -len(generated_tokens)]
            PROMPT_CACHE.insert_cache(
                model_name,
                prompt_only_key,
                prompt_only_cache,
                model_tokens=prompt_model_tokens_list,
            )
            SESSION_INDEX.register_cache_key(session_ctx, prompt_only_key)
        except Exception:
            pass

    PROMPT_CACHE.insert_cache(
        model_name, cache_key, prompt_cache, model_tokens=model_tokens_full
    )
    SESSION_INDEX.register_cache_key(session_ctx, cache_key)


def _vlm_cache_covers_partial_image(
    model_tokens: List[int], rest_tokens: List[int]
) -> Tuple[bool, int, int]:
    """
    True if the current cache cut leaves rest_tokens with a partial span of
    image-pad tokens (some but not all).  mlx-vlm's VLM forward demands that
    the count of image_pad tokens in input_ids exactly equals the total
    features in pixel_values — otherwise it raises:

        Image features and image tokens do not match: tokens: N, features M

    Callers use the bool to decide whether to retreat to a fresh prefill.
    Returns (is_partial, full_pad_count, rest_pad_count) so callers can log it.

    When the model has no resolvable image token id (mixed-arch VLMs), this
    function reports no partial — the _stream_generate_unified fallback
    handles those conservatively (use_vision only on cold miss).
    """
    if VLM_IMAGE_TOKEN_ID is None and VLM_VIDEO_TOKEN_ID is None:
        return False, 0, 0
    pad_ids = set()
    if VLM_IMAGE_TOKEN_ID is not None:
        pad_ids.add(int(VLM_IMAGE_TOKEN_ID))
    if VLM_VIDEO_TOKEN_ID is not None:
        pad_ids.add(int(VLM_VIDEO_TOKEN_ID))
    full_count = sum(1 for t in model_tokens if int(t) in pad_ids)
    if full_count == 0:
        return False, 0, 0
    rest_count = sum(1 for t in rest_tokens if int(t) in pad_ids)
    cached_pads = full_count - rest_count
    is_partial = 0 < cached_pads < full_count
    return is_partial, full_count, rest_count


def _kv_cache_offset(cache: Any) -> Optional[int]:
    """
    Return the current KV offset (number of original tokens actually stored in the
    KV cache) after any trim_prompt_cache() call.

    KVCache / QuantizedKVCache / RotatingKVCache all expose an int `.offset`.
    Hybrid VLM models (e.g. Qwen3.6 Qwen3_5MoeForConditionalGeneration) mix
    attention layers with linear/mamba layers that use ArraysCache, which has NO
    `.offset`.  In that case we must scan the layers for the first attention
    cache that does expose an integer offset — using layer[0] blindly would read
    an ArraysCache and return None, forcing the caller to fall back to the
    canonical matched_prefix_len (which is in canonical-token space, NOT model-
    token space, and lands mid image-pad expansion on VLM turns with images).

    Returns None only when NO layer exposes a usable offset.
    """
    if cache is None:
        return None
    layers = cache if isinstance(cache, (list, tuple)) else [cache]
    for layer in layers:
        off = getattr(layer, "offset", None)
        if off is None:
            continue
        try:
            return int(off)
        except (TypeError, ValueError):
            continue
    return None


def _assert_cache_key_safety(
    original_prompt: str,
    cache_key_prompt: str,
    context: str = "",
) -> bool:
    """
    Safety invariant check for the dual-pipeline architecture.
    Asserts that the cache key is not dramatically shorter than the original prompt,
    which would indicate over-matching normalization silently deleting content.

    Only called when CACHE_NORM_SAFETY_CHECK=true (off by default — no latency impact).

    Returns True if invariant holds.  On violation: logs an error and returns False.
    The caller must fall back to using original_prompt as the cache key.
    """
    if not isinstance(original_prompt, str) or not isinstance(cache_key_prompt, str):
        return True
    orig_len = len(original_prompt)
    if orig_len == 0:
        return True
    key_len = len(cache_key_prompt)
    ratio = key_len / orig_len
    if ratio < 0.90:
        _terminal_status(
            "❌",
            f"[CACHE-SAFETY] cache_key is {ratio:.1%} of original prompt "
            f"({key_len} vs {orig_len} chars) — normalization over-matched. "
            f"Falling back to original prompt as cache key. context={context}",
        )
        return False
    return True


def _scrub_cache_key(prompt: str) -> str:
    """
    Post-render scrub of the CACHE KEY ONLY — never called on the model input.

    Applies only atomic, line-scoped or structurally-bounded substitutions that
    cannot span section boundaries.  Volatile-but-semantically-neutral tokens:
    - "Current time is …"         → __CACHE_STABLE_TIME__
    - cch=<hex>;                  → cch=STATIC_CACHE;
    - -anthropic-billing-header:  → STATIC_CACHE value
    - <system-reminder>…</system-reminder>  → __CACHE_STABLE_SYSTEM_REMINDER__

    The Inbound Context block and message_id scrubbing are handled upstream by
    _canonicalize_messages() (message-struct level, before rendering).

    Returns the original string unchanged if cache_canonicalize_tool_context is off
    or the input is not a string.
    """
    if not isinstance(prompt, str):
        return prompt
    if not SETTINGS.cache_canonicalize_tool_context:
        return prompt
    # Scrub timestamp and Claude Code telemetry so retries don't break cache.
    normalized = CACHE_TIME_PATTERN.sub("__CACHE_STABLE_TIME__", prompt)
    normalized = CACHE_CCH_PATTERN.sub("cch=STATIC_CACHE;", normalized)
    normalized = CACHE_BILLING_HEADER_PATTERN.sub(
        "-anthropic-billing-header: STATIC_CACHE", normalized
    )
    normalized = CACHE_SYSTEM_REMINDER_PATTERN.sub(
        "__CACHE_STABLE_SYSTEM_REMINDER__", normalized
    )
    return normalized


# _normalize_prompt_for_cache alias removed (Phase 7 Fix 4, 2026-03-14).
# No call sites remain after Phase 6 cleanup.  Use _scrub_cache_key() directly.


# --- DUAL-PIPELINE: message-struct level canonicalization ---
# Structural anchors for the OpenClaw "Inbound Context (trusted metadata)" block.
# These constants are intentionally plain strings — no regex — so there is no
# risk of accidental over-matching on user content.
_INBOUND_CONTEXT_HEADER = (
    "## Group Chat Context\n## Inbound Context (trusted metadata)\n"
)
_INBOUND_CONTEXT_FENCE_OPEN = "```json"
_INBOUND_CONTEXT_FENCE_CLOSE = "```"
# Stable sentinel that replaces the ENTIRE Group Chat Context / Inbound Context
# block (header + description lines + JSON fence) in the canonical cache key.
# Using a single sentinel for the whole block — instead of replacing only the
# JSON payload — ensures canonical form is IDENTICAL whether or not OpenClaw
# included the block in a given turn.  OpenClaw sometimes omits the block
# (confirmed in Phase 8 session logs: 44059 chars in T2, 43478 chars in T7),
# which caused the entire KV cache to diverge at the omission point and produced
# a 50.4% cache hit instead of the expected ~98%.
_INBOUND_CONTEXT_STABLE_SECTION = "__STABLE_INBOUND_CONTEXT_SECTION__"
# Anchor used for the "block absent" injection path.
_PROJECT_CONTEXT_ANCHOR = "\n# Project Context"


def _canonicalize_inbound_context_block(content: str) -> str:
    """
    Canonicalize the OpenClaw "Group Chat Context / Inbound Context" section.

    Two cases are handled so the canonical form is IDENTICAL regardless of whether
    OpenClaw included the block in a given turn (Phase 8 finding: the block is
    sometimes present, sometimes absent, causing a 50.4% VLM cache hit instead of
    the expected ~98% when the block is omitted in one turn but was present in the
    turn whose KV state is cached).

    Case A — block IS present:
        Replace the ENTIRE block (from the '## Group Chat Context' line through
        the closing ``` fence) with _INBOUND_CONTEXT_STABLE_SECTION.
        Canonical: '...<before>\\n__STABLE_INBOUND_CONTEXT_SECTION__\\n# Project Context...'

    Case B — block is ABSENT but '\\n# Project Context' anchor exists:
        Insert _INBOUND_CONTEXT_STABLE_SECTION immediately before the anchor.
        Canonical: same as Case A.

    If neither condition is met, the content is returned unchanged (safe no-op).

    Safety guarantees:
    - Pure string operations, no regex, no DOTALL.
    - Only touches the bounded region between the header and closing ```.
    - Stops scanning after 10 description lines (cannot cross section boundaries).
    - Idempotent: already-stable content is returned unchanged in one fast check.
    """
    # Fast idempotency check.
    if _INBOUND_CONTEXT_STABLE_SECTION in content:
        return content

    header_pos = content.find(_INBOUND_CONTEXT_HEADER)

    if header_pos != -1:
        # --- Case A: block present ---
        # Find the start of the '## Group Chat Context' line.
        # header_pos already points there since _INBOUND_CONTEXT_HEADER begins with it.
        # Walk forward (up to 10 lines) to find the opening ```json fence.
        search_start = header_pos + len(_INBOUND_CONTEXT_HEADER)
        fence_open_pos = -1
        cursor = search_start
        for _ in range(10):
            line_end = content.find("\n", cursor)
            if line_end == -1:
                break
            line = content[cursor:line_end]
            if line.startswith(_INBOUND_CONTEXT_FENCE_OPEN):
                fence_open_pos = cursor
                break
            # Stop if a new section header is encountered — prevents over-scanning.
            if line.startswith("## ") or line.startswith("# "):
                break
            cursor = line_end + 1

        if fence_open_pos == -1:
            # JSON fence not found within 10 lines — leave unchanged (safe).
            return content

        # Find the closing ``` fence on its own line.
        fence_line_end = content.find("\n", fence_open_pos)
        if fence_line_end == -1:
            return content
        json_body_start = fence_line_end + 1
        fence_close_pos = content.find(
            "\n" + _INBOUND_CONTEXT_FENCE_CLOSE, json_body_start
        )
        if fence_close_pos == -1:
            return content
        # fence_close_pos + 1 = first ` of the closing fence.
        # We advance past the full closing fence line (```\n) to get the remainder.
        fence_end = fence_close_pos + 1 + len(_INBOUND_CONTEXT_FENCE_CLOSE)
        # Skip the trailing newline if present.
        if fence_end < len(content) and content[fence_end] == "\n":
            fence_end += 1

        # Determine prefix boundary: keep the content immediately before the block.
        # header_pos may be preceded by a newline we want to preserve.
        block_start = header_pos

        return (
            content[:block_start]
            + _INBOUND_CONTEXT_STABLE_SECTION
            + "\n"
            + content[fence_end:]
        )

    else:
        # --- Case B: block absent but # Project Context anchor present ---
        # OpenClaw sometimes omits the Inbound Context section entirely.
        # Insert the stable sentinel at the same relative position so the cache key
        # matches turns where the block was present (Case A canonical form).
        anchor_pos = content.find(_PROJECT_CONTEXT_ANCHOR)
        if anchor_pos == -1:
            return content  # No recognisable anchor — safe no-op.
        # Insert stable sentinel on its own line immediately before the anchor.
        # _PROJECT_CONTEXT_ANCHOR starts with '\n', so after anchor_pos we have
        # '\n# Project Context'. We insert before that newline.
        return (
            content[:anchor_pos]
            + "\n"
            + _INBOUND_CONTEXT_STABLE_SECTION
            + content[anchor_pos:]
        )


def _canonicalize_messages(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Pre-render canonicalization: returns (original_messages, canonical_messages).

    - original_messages  : deep copy of the input, never modified.
      This is what the model always receives (fed to apply_chat_template for generation).
    - canonical_messages : deep copy with volatile-but-semantically-neutral fields
      replaced by stable tokens.  Used ONLY for cache key computation.

    Volatile fields normalised in the canonical copy (message-struct level only):
    1. "message_id" values inside JSON content strings → __STABLE_MSG_ID__
       (handles OpenClaw / Claude Code per-request IDs that change every turn)
    2. OpenClaw "Inbound Context (trusted metadata)" JSON block → __STABLE_INBOUND_META__
       (handled structurally via string anchors, no DOTALL regex)

    No other mutations.  In particular, tool results and <think> blocks are
    preserved verbatim — the model sees them intact.
    """
    if not SETTINGS.cache_canonicalize_tool_context:
        # Canonicalization disabled: both pipelines see the same content.
        original = copy.deepcopy(messages)
        return original, copy.deepcopy(original)

    original: List[Dict[str, Any]] = copy.deepcopy(messages)
    canonical: List[Dict[str, Any]] = copy.deepcopy(messages)

    for msg in canonical:
        content = msg.get("content", "")
        if isinstance(content, str):
            # 1. Stable message IDs.
            content = INBOUND_META_MESSAGE_ID_PATTERN.sub(
                r"\1__STABLE_MSG_ID__\2", content
            )
            # 2. Inbound Context block (system message — present or absent).
            content = _canonicalize_inbound_context_block(content)
            # 3. Sub-agent completion stats (volatile runtime duration in user messages).
            content = SUBAGENT_STATS_PATTERN.sub(r"\1__STABLE_RUNTIME__", content)
            msg["content"] = content
        elif isinstance(content, list):
            new_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")
                    text = INBOUND_META_MESSAGE_ID_PATTERN.sub(
                        r"\1__STABLE_MSG_ID__\2", text
                    )
                    text = _canonicalize_inbound_context_block(text)
                    text = SUBAGENT_STATS_PATTERN.sub(r"\1__STABLE_RUNTIME__", text)
                    new_parts.append({**part, "text": text})
                else:
                    new_parts.append(part)
            msg["content"] = new_parts

    return original, canonical


def _extract_session_context(
    body: Dict[str, Any], prompt_tokens: List[int]
) -> SessionContext:
    def _read_any_id(
        container: Optional[Dict[str, Any]], keys: List[str]
    ) -> Optional[str]:
        if not isinstance(container, dict):
            return None
        for key in keys:
            value = container.get(key)
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (int, float)):
                return str(value)
        return None

    metadata = body.get("metadata")
    extra_body = body.get("extra_body")

    id_keys = [
        "session_id",
        "conversation_id",
        "thread_id",
        "chat_id",
        "conversation",
        "session",
    ]
    parent_keys = [
        "parent_session_id",
        "parent_id",
        "source_session_id",
        "origin_session_id",
    ]
    branch_keys = ["branch_id", "branch", "subsession_id", "subagent_id"]

    session_id = (
        _read_any_id(body, id_keys)
        or _read_any_id(metadata, id_keys)
        or _read_any_id(extra_body, id_keys)
    )
    parent_session_id = (
        _read_any_id(body, parent_keys)
        or _read_any_id(metadata, parent_keys)
        or _read_any_id(extra_body, parent_keys)
    )
    branch_id = (
        _read_any_id(body, branch_keys)
        or _read_any_id(metadata, branch_keys)
        or _read_any_id(extra_body, branch_keys)
    )

    source = "request"
    if not session_id:
        raw = ",".join(str(tok) for tok in prompt_tokens[:128])
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
        session_id = f"implicit-{digest}"
        source = "derived_from_prompt_prefix"

    if parent_session_id == session_id:
        parent_session_id = None

    return SessionContext(
        session_id=session_id,
        parent_session_id=parent_session_id,
        branch_id=branch_id,
        source=source,
    )


def _build_sampler(body):
    """
    Build a sampler with conservative anti-loop defaults.
    If mlx_lm in this environment does not support some kwargs, fall back safely.
    """
    temperature = body.get("temperature", SETTINGS.default_temperature)
    if not isinstance(temperature, (int, float)):
        temperature = 0.1
    if temperature < 0.01:
        temperature = 0

    # Anti-loop defaults can be overridden by request body.
    candidate_kwargs = {
        "temp": temperature,
        "top_p": body.get("top_p", SETTINGS.default_top_p),
        "top_k": body.get("top_k", SETTINGS.default_top_k),
        "min_p": body.get("min_p", SETTINGS.default_min_p),
        "repetition_penalty": body.get(
            "repetition_penalty", SETTINGS.default_repetition_penalty
        ),
        "repetition_context_size": body.get(
            "repetition_context_size", SETTINGS.default_repetition_context_size
        ),
    }

    # Some mlx_lm versions don't support all params. Drop unsupported keys progressively.
    kwargs = dict(candidate_kwargs)
    while True:
        try:
            return make_sampler(**kwargs), kwargs
        except TypeError as e:
            msg = str(e)
            removed = False
            for key in list(kwargs.keys()):
                # Typical error: unexpected keyword argument 'xyz'
                if f"'{key}'" in msg:
                    kwargs.pop(key, None)
                    removed = True
                    break
            if not removed:
                # Unknown failure mode, use minimal safe sampler.
                return make_sampler(temp=temperature), {"temp": temperature}


def _find_pids_listening_on_port(port: int) -> List[int]:
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if result.returncode not in (0, 1):
        return []
    pids: List[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def _stop_stale_litellm_on_proxy_port(port: int) -> None:
    pids = _find_pids_listening_on_port(port)
    if not pids:
        return
    stopped_any = False
    for pid in pids:
        cmdline = ""
        try:
            ps_result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "command="],
                check=False,
                capture_output=True,
                text=True,
            )
            cmdline = ps_result.stdout.strip().lower()
        except Exception:
            pass
        if "litellm" not in cmdline:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            stopped_any = True
            time.sleep(0.2)
            try:
                os.kill(pid, 0)
            except OSError:
                continue
            os.kill(pid, signal.SIGKILL)
        except OSError:
            continue
    if stopped_any:
        _terminal_status(
            "♻️", f"Stopped stale LiteLLM process(es) on port {port} before restart."
        )


def start_litellm_proxy():
    global proxy_process, proxy_config_path
    _stop_stale_litellm_on_proxy_port(SETTINGS.proxy_port)
    _terminal_status("🌉", f"Launching LiteLLM Proxy on port {SETTINGS.proxy_port}...")

    # Use proxy config so unsupported OpenAI params (e.g. "store") are dropped.
    # request_timeout (seconds): allow long prefills (e.g. 75k tokens) so client retries
    # don't trigger a timeout death spiral; 20 min is generous for local MLX.
    config_yaml = f"""model_list:
  - model_name: {SETTINGS.proxy_model_id}
    litellm_params:
      model: {SETTINGS.proxy_model_id}
      api_base: http://127.0.0.1:{SETTINGS.mlx_port}/v1
      api_key: local
      timeout: 1200
      # Keep these request fields when drop_params=true so this server can map
      # OpenClaw/Claude reasoning intent into tokenizer enable_thinking.
      allowed_openai_params:
        - reasoning_effort
litellm_settings:
  drop_params: true
  request_timeout: 1200
"""
    fd, temp_path = tempfile.mkstemp(prefix="litellm-qwen-", suffix=".yaml")
    with os.fdopen(fd, "w") as config_file:
        config_file.write(config_yaml)
    proxy_config_path = temp_path

    litellm_cli = Path(sys.executable).with_name("litellm")
    if litellm_cli.exists():
        cmd = [
            str(litellm_cli),
            "--config",
            proxy_config_path,
            "--port",
            str(SETTINGS.proxy_port),
        ]
    else:
        # Fallback path for environments where the CLI script is not next to Python.
        resolved = shutil.which("litellm")
        if resolved:
            cmd = [
                resolved,
                "--config",
                proxy_config_path,
                "--port",
                str(SETTINGS.proxy_port),
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "litellm",
                "--config",
                proxy_config_path,
                "--port",
                str(SETTINGS.proxy_port),
            ]

    my_env = os.environ.copy()
    my_env["OPENAI_API_KEY"] = "local"
    proxy_process = subprocess.Popen(
        cmd, env=my_env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    time.sleep(SETTINGS.proxy_startup_wait_seconds)
    if proxy_process.poll() is not None:
        stderr_output = ""
        if proxy_process.stderr is not None:
            try:
                stderr_output = proxy_process.stderr.read().decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                stderr_output = ""
        raise RuntimeError(
            "LiteLLM proxy failed to start. "
            + (
                f"stderr: {stderr_output.strip()}"
                if stderr_output
                else "No stderr captured."
            )
        )
    _terminal_status(
        "✅",
        f"LiteLLM Proxy ready at http://127.0.0.1:{SETTINGS.proxy_port} (model: {SETTINGS.proxy_model_id})",
    )


def cleanup():
    global proxy_config_path
    if proxy_process:
        _terminal_status("🧹", "Shutting down LiteLLM Proxy...")
        proxy_process.terminate()
        proxy_process.wait()
    if proxy_config_path:
        try:
            Path(proxy_config_path).unlink(missing_ok=True)
        except Exception:
            pass
    _terminal_status("👋", "MLX Server stopped.")


atexit.register(cleanup)

# Model loading: LM or VLM based on config.
model = None
tokenizer = None
processor = None
is_vlm = False
vlm_config = None

_terminal_status(
    "🚀",
    f"Loading model: {SETTINGS.model_path} (exposed as: {SETTINGS.proxy_model_id})",
)
_resolved_path, _config = _resolve_model_path_and_config()
if _config and _is_vlm_config(_config):
    if not mlx_vlm_available:
        raise RuntimeError(
            "Model config indicates a VLM but mlx-vlm is not installed. "
            "Install with: pip install mlx-vlm (and optionally mlx-vlm[torch] for some models)."
        )
    # Workaround: when torchvision is missing, transformers sets VIDEO_PROCESSOR_MAPPING_NAMES
    # values to None, then video_processor_class_from_name does "class_name in extractors"
    # and raises TypeError. Patch to skip None so processor can load (image-only path).
    try:
        import importlib
        from transformers.models.auto import video_processing_auto as _vpa
        from transformers.models.auto.configuration_auto import (
            model_type_to_module_name,
        )

        def _patched_video_processor_class_from_name(class_name: str):
            for module_name, extractors in _vpa.VIDEO_PROCESSOR_MAPPING_NAMES.items():
                if extractors is not None and class_name in extractors:
                    mod_name = model_type_to_module_name(module_name)
                    module = importlib.import_module(
                        f".{mod_name}", "transformers.models"
                    )
                    try:
                        return getattr(module, class_name)
                    except AttributeError:
                        continue
            for extractor in _vpa.VIDEO_PROCESSOR_MAPPING._extra_content.values():
                if getattr(extractor, "__name__", None) == class_name:
                    return extractor
            main_module = importlib.import_module("transformers")
            if hasattr(main_module, class_name):
                return getattr(main_module, class_name)
            return None

        _vpa.video_processor_class_from_name = _patched_video_processor_class_from_name
    except Exception:
        pass
    model, processor = load_vlm(
        SETTINGS.model_path, tokenizer_config={"trust_remote_code": True}
    )
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    is_vlm = True
    vlm_config = (
        _config if isinstance(_config, dict) else getattr(model, "config", None)
    )
    if vlm_config is None and hasattr(model, "config"):
        vlm_config = getattr(model.config, "__dict__", None) or {}
    _terminal_status("✅", "VLM model loaded (mlx-vlm).")
    # Resolve the model's image / video pad token IDs once, after config is loaded.
    # Used by _stream_generate_unified to decide (safely) whether rest_tokens
    # still contains image placeholders that need pixel_values, and by the VLM
    # cache safety check to detect mid-image cache cuts.
    VLM_IMAGE_TOKEN_ID, VLM_VIDEO_TOKEN_ID = _resolve_vlm_image_token_ids(
        vlm_config if isinstance(vlm_config, dict) else (_config or {})
    )
    _terminal_status(
        "🖼️",
        f"VLM image_token_id={VLM_IMAGE_TOKEN_ID} video_token_id={VLM_VIDEO_TOKEN_ID}",
    )
    try:
        from transformers.utils import is_torchvision_available

        if is_torchvision_available():
            _terminal_status(
                "⚡",
                "Torch acceleration: enabled (fast image/video processor). "
                "If you see Metal 'uncommitted encoder' crash, try: pip install mlx-vlm (without [torch] extra).",
            )
        else:
            _terminal_status(
                "⚠️",
                "Torch acceleration: disabled. Install mlx-vlm[torch] for much faster image processing.",
            )
    except Exception:
        _terminal_status(
            "⚠️", "Torch acceleration: unknown (torch/torchvision not detected)."
        )
else:
    model, tokenizer = load(
        SETTINGS.model_path, tokenizer_config={"trust_remote_code": True}
    )
    _terminal_status("✅", "Model loaded (mlx-lm).")
    _terminal_status("⚡", "Torch acceleration: N/A (text-only model).")

# P6/1e: probe whether `preserve_thinking` can be forwarded to the active
# tokenizer / processor's `apply_chat_template`.  Done once, here, after
# model load so both LM and VLM target sets are populated.  Result caches
# in `_PRESERVE_THINKING_SUPPORTED`; `_chat_template_extras` only forwards
# the kwarg when True.  Guards against a future transformers / mlx-vlm
# upgrade that tightens kwarg checking — pre-P6 forwarded unconditionally
# and would TypeError every request after such an upgrade.
_probe_targets = [tokenizer]
if is_vlm and processor is not None:
    _probe_targets.append(processor)
    _maybe_proc_tok = getattr(processor, "tokenizer", None)
    if _maybe_proc_tok is not None and _maybe_proc_tok is not tokenizer:
        _probe_targets.append(_maybe_proc_tok)
try:
    _PRESERVE_THINKING_SUPPORTED = _probe_preserve_thinking_support(_probe_targets)
except Exception:
    _PRESERVE_THINKING_SUPPORTED = False
_terminal_status(
    "🧵",
    f"preserve_thinking probe: "
    f"{'supported' if _PRESERVE_THINKING_SUPPORTED else 'NOT supported — kwarg will not be forwarded'}",
)

_terminal_status(
    "🧠",
    (
        "Global Cache Config loaded | "
        f"max_entries={SETTINGS.prompt_cache_max_entries_global} | "
        f"ttl_seconds={SETTINGS.prompt_cache_ttl_seconds} | "
        f"canonicalize_tool_context={SETTINGS.cache_canonicalize_tool_context} | "
        f"healing_store_capacity={MAX_HEALING_STORE}"
    ),
)
_kv_desc = f"bits={SETTINGS.kv_bits}" if SETTINGS.kv_bits is not None else "OFF"
if SETTINGS.kv_bits is not None and SETTINGS.kv_quant_scheme == "turboquant":
    _kv_desc += f" scheme=turboquant start={SETTINGS.quantized_kv_start}"
elif SETTINGS.kv_bits is not None:
    _kv_desc += f" scheme=uniform group_size={SETTINGS.kv_group_size}"
_terminal_status("🗜️", f"KV Cache Quantization: {_kv_desc}")


def _stream_generate_kwargs(prompt_tokens, max_tokens, sampler, prompt_cache):
    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": prompt_tokens,
        "max_tokens": max_tokens,
        "sampler": sampler,
        "prompt_cache": prompt_cache,
        "max_kv_size": SETTINGS.max_kv_size,
        "kv_group_size": SETTINGS.kv_group_size,
    }
    if SETTINGS.kv_bits is not None:
        kwargs["kv_bits"] = SETTINGS.kv_bits
    return kwargs


def _stream_generate_unified(
    rest_tokens,
    max_tokens,
    sampler,
    prompt_cache,
    vlm_pixel_values=None,
    vlm_mask=None,
    vlm_kwargs=None,
    cache_match_type="miss",
):
    """
    Yields response objects with .text and .token (LM: GenerationResponse, VLM: GenerationResult).
    Dynamically checks for image tokens in rest_tokens to prevent Metal Segmentation Faults.
    """
    if is_vlm:
        # Pass vision tensors ONLY when rest_tokens still contains image_pad
        # placeholders that mlx-vlm needs to fill with image embeddings.  If
        # the cache already covers the image span, rest_tokens has zero image
        # pads and pixel_values MUST be None — otherwise mlx-vlm's safety
        # check raises "Image features and image tokens do not match:
        # tokens: N, features M" when N != full image feature count.
        image_pad_ids = set()
        if VLM_IMAGE_TOKEN_ID is not None:
            image_pad_ids.add(int(VLM_IMAGE_TOKEN_ID))
        if VLM_VIDEO_TOKEN_ID is not None:
            image_pad_ids.add(int(VLM_VIDEO_TOKEN_ID))

        rest_image_pad_count = 0
        if rest_tokens and image_pad_ids:
            rest_image_pad_count = sum(
                1 for tok in rest_tokens if int(tok) in image_pad_ids
            )
        # Fallback for unknown VLMs: be conservative — on a cold miss, pass
        # pixel_values; on any cache hit, trust that the prior turn's KV
        # already absorbed the image and skip pixel_values.
        if not image_pad_ids:
            has_image_tokens_fallback = cache_match_type == "miss"
            use_vision = has_image_tokens_fallback and vlm_pixel_values is not None
        else:
            use_vision = (
                rest_image_pad_count > 0 and vlm_pixel_values is not None
            )

        rest_ids = (
            mx.array([rest_tokens], dtype=mx.int32)
            if rest_tokens
            else mx.array([[0]], dtype=mx.int32)
        )

        sliced_mask = None
        if vlm_mask is not None:
            if rest_tokens:
                sliced_mask = vlm_mask[..., -len(rest_tokens) :]
            else:
                sliced_mask = vlm_mask[..., -1:]

        kwargs = {
            "input_ids": rest_ids,
            "pixel_values": vlm_pixel_values if use_vision else None,
            "mask": sliced_mask if use_vision else None,
            **(vlm_kwargs if vlm_kwargs else {}),
            "prompt_cache": prompt_cache,
            "max_tokens": max_tokens,
            "sampler": sampler,
            "max_kv_size": SETTINGS.max_kv_size,
            "kv_group_size": SETTINGS.kv_group_size,
            "kv_quant_scheme": SETTINGS.kv_quant_scheme,
            "quantized_kv_start": SETTINGS.quantized_kv_start,
        }
        if SETTINGS.kv_bits is not None:
            kwargs["kv_bits"] = SETTINGS.kv_bits
        for resp in stream_generate_vlm(model, processor, "", image=None, **kwargs):
            yield resp
    else:
        for resp in stream_generate(
            **_stream_generate_kwargs(rest_tokens, max_tokens, sampler, prompt_cache)
        ):
            yield resp


class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep terminal output focused on custom request lifecycle lines.
        return

    def _mark_connection_close(self) -> None:
        # Single-threaded HTTPServer (see run()): HTTP/1.1 keep-alive lets one
        # client (e.g. LiteLLM's connection pool) block every other client
        # forever because handle_one_request() loops on the same socket until
        # the peer closes.  Forcing close after every response keeps the
        # single-threaded accept loop fair — each request is served on its own
        # connection without starving anyone.
        try:
            self.close_connection = True
        except Exception:
            pass

    def do_GET(self):
        self._mark_connection_close()

        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": SETTINGS.proxy_model_id,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "mlx",
                            }
                        ],
                    }
                ).encode("utf-8")
            )
            return
        self.send_error(404, "Not Found")

    def do_POST(self):
        self._mark_connection_close()
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self.do_GET()

        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self.send_error(404, "Not Found")
            return


        try:
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except Exception:
            self.send_error(400, "Bad Request")
            return

        request_id = uuid.uuid4().hex[:12]
        # P4: publish request_id to thread-local so cache-layer bump sites
        # (inside LRUPromptCache, inside `_trim_to_model_depth`) can
        # attribute to this request without threading request_id through
        # every internal call.  Cleared in the outer finally.
        _set_current_request_id(request_id)
        tools = body.get("tools")
        reasoning_control = _extract_enable_thinking(body)
        enable_thinking = reasoning_control["enable_thinking"]

        vlm_pixel_values = None
        vlm_mask = None
        vlm_input_ids_raw = None
        vlm_kwargs = None

        if is_vlm:
            raw_messages = body.get("messages", [])
            # --- APPLY STATELESS HEALING ---
            healed_messages = _heal_messages(raw_messages)

            messages = _prepare_messages_for_vlm(healed_messages, tools=tools)
            images = _extract_images_from_messages(body.get("messages", []))

            with model_lock:
                vlm_input_ids_raw, vlm_pixel_values, vlm_mask, vlm_kwargs = (
                    _vlm_prompt_and_inputs(
                        processor,
                        vlm_config or {},
                        messages,
                        images,
                        tools=tools,
                        enable_thinking=enable_thinking,
                        **{
                            k: v
                            for k, v in _chat_template_extras(enable_thinking).items()
                            if k != "enable_thinking"
                        },
                    )
                )
                _vlm_sync_before_generation(vlm_pixel_values, vlm_mask)

            if vlm_input_ids_raw is not None:
                if hasattr(vlm_input_ids_raw, "flatten"):
                    model_tokens = vlm_input_ids_raw.flatten().tolist()
                else:
                    model_tokens = list(vlm_input_ids_raw)
            else:
                model_tokens = []

            # --- VLM DUAL PIPELINE: canonical cache key (Phase 8) ---
            # The model always prefills from model_tokens (original messages, above).
            # For the cache lookup we apply the same canonicalization as the LM path:
            # _canonicalize_messages + _scrub_cache_key.  This ensures volatile fields
            # in the system message (Inbound Context block, message IDs, etc.) do not
            # cause cache divergence when OpenClaw changes them between turns.
            # We use the VLM tokenizer's text-only apply_chat_template (CPU, no GPU
            # required) so the canonical key is derived without a second model_lock
            # acquisition.  Falls back to model_tokens if tokenizer is unavailable.
            prompt_tokens = model_tokens  # safe fallback
            if SETTINGS.cache_canonicalize_tool_context:
                try:
                    _, canonical_msgs_vlm = _canonicalize_messages(healed_messages)
                    canon_prepared = _prepare_messages_for_vlm(
                        canonical_msgs_vlm, tools=tools
                    )
                    # Resolve the text-only tokenizer from the VLM processor.
                    _vlm_tok = None
                    if processor is not None:
                        if (
                            hasattr(processor, "tokenizer")
                            and processor.tokenizer is not None
                            and hasattr(processor.tokenizer, "apply_chat_template")
                            and getattr(processor.tokenizer, "chat_template", None)
                        ):
                            _vlm_tok = processor.tokenizer
                        elif hasattr(processor, "apply_chat_template") and getattr(
                            processor, "chat_template", None
                        ):
                            _vlm_tok = processor
                    if _vlm_tok is not None:
                        _canon_fmt = _vlm_tok.apply_chat_template(
                            canon_prepared,
                            tokenize=False,
                            add_generation_prompt=True,
                            tools=tools,
                            **_chat_template_extras(enable_thinking),
                        )
                        _canon_fmt = _scrub_cache_key(str(_canon_fmt))
                        _canon_ids = _vlm_tok.encode(_canon_fmt)
                        if isinstance(_canon_ids, list) and _canon_ids:
                            # G1: make canonical cache key image-content
                            # aware.  Without this, `<|image_pad|>` is a
                            # single token that encodes identically for any
                            # image payload — two requests with the same
                            # message structure but different images
                            # produce byte-identical canonical keys and
                            # silently reuse each other's KV attention.
                            # Markers live in canonical space ONLY; the
                            # model path (model_tokens) is untouched.
                            if (
                                SETTINGS.cache_vlm_image_identity
                                and VLM_IMAGE_TOKEN_ID is not None
                                and images
                            ):
                                _img_markers = _image_identity_markers(images)
                                if _img_markers:
                                    _canon_ids = _inject_image_markers(
                                        _canon_ids,
                                        _img_markers,
                                        int(VLM_IMAGE_TOKEN_ID),
                                    )
                            prompt_tokens = _canon_ids
                except Exception:
                    pass  # Fall back to model_tokens — correct but no canonicalization

            prompt = ""

        else:
            raw_messages = body.get("messages", [])
            # --- APPLY STATELESS HEALING ---
            healed_messages = _heal_messages(raw_messages)

            # --- DUAL PIPELINE: split model input from cache key at message-struct level ---
            # original_messages → rendered → model sees this (never normalized)
            # canonical_messages → rendered → scrubbed → cache lookup key only
            original_messages, canonical_messages = _canonicalize_messages(
                healed_messages
            )
            messages = _prepare_messages_for_template(original_messages)
            cache_messages = _prepare_messages_for_template(canonical_messages)

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                    **_chat_template_extras(enable_thinking),
                )
                cache_prompt_raw = tokenizer.apply_chat_template(
                    cache_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    tools=tools,
                    **_chat_template_extras(enable_thinking),
                )
            else:
                prompt = messages[-1]["content"] if messages else ""
                cache_prompt_raw = (
                    cache_messages[-1]["content"] if cache_messages else ""
                )

            # Post-render scrub: atomic, line-scoped patterns only (cch=, billing header,
            # timestamps, system-reminder). Never applied to the model input.
            cache_prompt = _scrub_cache_key(cache_prompt_raw)
            # Safety invariant: cache key must not be dramatically shorter than the original.
            if SETTINGS.cache_norm_safety_check:
                if not _assert_cache_key_safety(
                    prompt, cache_prompt, context="lm_path"
                ):
                    # Normalization over-matched — fall back to using the original prompt
                    # as the cache key to prevent invisible content loss.
                    cache_prompt = prompt
            # Cache lookup uses canonical token sequence; model input uses original tokens.
            prompt_tokens = _tokenize_prompt(cache_prompt)
            model_tokens = _tokenize_prompt(prompt)
            prompt_was_normalized = cache_prompt != prompt
            cache_key_delta_chars = len(prompt) - len(cache_prompt)

        session_ctx = _extract_session_context(body, prompt_tokens)
        if is_vlm:
            # prompt_was_normalized: True when canonical pipeline fired and produced
            # different token count from model_tokens (i.e. canonicalization changed
            # something). When the canonical pipeline falls back to model_tokens (no
            # tokenizer available or canonicalization disabled), prompt_tokens ==
            # model_tokens and we report no normalization.
            prompt_was_normalized = bool(
                SETTINGS.cache_canonicalize_tool_context
                and prompt_tokens is not model_tokens
                and len(prompt_tokens) != len(model_tokens)
            )
            cache_key_delta_chars = len(model_tokens) - len(prompt_tokens)
        cache_key = prompt_tokens[:]
        prompt_cache = None
        # model_tokens: tokens derived from the original (unmodified) prompt.
        # The model always prefills from this sequence, never from the canonical cache key.
        rest_tokens = model_tokens
        cache_session_tokens = prompt_tokens
        cache_match_type = "miss"
        matched_prefix_len = 0
        cache_selection_source = "none"
        rest_count = len(model_tokens)
        request_logger = None
        sampler, sampler_kwargs = _build_sampler(body)
        max_tokens = body.get("max_tokens", SETTINGS.default_max_tokens)
        is_streaming = body.get("stream", False)

        acquired = False
        generated_tokens = []
        message_text = ""
        generation_started_at = None
        first_token_at = None
        queue_started_at = time.time()

        # --- STABLE-PREFIX TELEMETRY DEFAULTS ---
        stable_prefix_token_len_computed = 0
        stable_prefix_msg_count_computed = 0
        stable_prefix_diff_descriptors: List[Dict[str, Any]] = []

        # --- G2 WATCHDOG STATE ---
        # Cooperative abort flag.  Set by a threading.Timer after
        # `generation_watchdog_seconds` of active generation to recover
        # `model_lock` from pathological decodes / stalls.  Yield loops
        # check is_set() after each token; finally cancels the timer.
        watchdog_flag = threading.Event()
        watchdog_timer: Optional[threading.Timer] = None
        generation_aborted = False

        try:
            model_lock.acquire(blocking=True)
            acquired = True

            generation_started_at = time.time()
            wait_seconds = generation_started_at - queue_started_at

            # Start the wall-clock deadline AFTER lock acquire so queue-wait
            # does not consume the generation budget.  A value of 0 disables
            # the watchdog (legacy behaviour — unbounded generation).
            _wd_s = SETTINGS.generation_watchdog_seconds
            if _wd_s and _wd_s > 0:
                watchdog_timer = threading.Timer(float(_wd_s), watchdog_flag.set)
                watchdog_timer.daemon = True
                watchdog_timer.start()
            with prompt_cache_lock:
                # --- M2: Compute stable prefix from message-level diff ---
                _session_id_for_turn = (session_ctx.session_id or "").strip()
                if _session_id_for_turn:
                    (
                        stable_prefix_token_len_computed,
                        stable_prefix_msg_count_computed,
                        stable_prefix_diff_descriptors,
                    ) = _stable_prefix_token_len(_session_id_for_turn, messages)

                # --- Standard global cache lookup ---
                # Cache lookup uses the canonical `prompt_tokens` for block-
                # hash candidate discovery AND the original `model_tokens`
                # for model-space LCP verification + KV trim.  The returned
                # `rest_model_tokens` is already sliced to the tail of
                # `model_tokens` that still needs prefilling — no post-hoc
                # math against _kv_cache_offset needed.  This is the
                # dual-pipeline correctness guarantee for VLM: canonical
                # prefix collisions (e.g. system-prompt reuse) cannot yield
                # KV states built from a different image-expansion sequence.
                (
                    prompt_cache,
                    rest_model_tokens,
                    cache_session_tokens,
                    cache_match_type,
                    matched_prefix_len,
                    cache_selection_source,
                ) = SESSION_INDEX.select_best_cache(
                    model_name=SETTINGS.model_path,
                    prompt_tokens=prompt_tokens,
                    session_ctx=session_ctx,
                    prompt_cache_store=PROMPT_CACHE,
                    model_tokens=model_tokens,
                )
                rest_tokens = (
                    rest_model_tokens
                    if rest_model_tokens is not None
                    else list(model_tokens)
                )
                _kv_off = _kv_cache_offset(prompt_cache)

                # Defensive VLM safety net: if LCP somehow landed mid-image
                # (same image-pad token id across turns but e.g. a missing
                # gen-tokens storage path), mlx-vlm's forward would raise
                # "Image features and image tokens do not match".  Retreat to
                # a fresh prefill rather than crash the request.
                if is_vlm and prompt_cache is not None:
                    _is_partial, _full_pads, _rest_pads = (
                        _vlm_cache_covers_partial_image(model_tokens, rest_tokens)
                    )
                    if _is_partial:
                        if SETTINGS.vlm_cache_debug:
                            _terminal_status(
                                "🛟",
                                f"[vlm-cache-retreat] cache cut mid-image "
                                f"(full_pads={_full_pads} rest_pads={_rest_pads}) — "
                                "dropping cache and prefilling fresh to satisfy mlx-vlm "
                                "image-feature parity.",
                                indent=1,
                            )
                        # 1a: evict the offending entry so it does not
                        # re-win candidate selection next turn and trap us
                        # in a retreat-forever starvation loop.
                        try:
                            if cache_session_tokens:
                                PROMPT_CACHE.delete(
                                    SETTINGS.model_path, cache_session_tokens
                                )
                        except Exception:
                            pass
                        # P4: track G1 effectiveness.  With G1 active this
                        # should never fire on clean traffic.  Any bump is
                        # a signal that G1's marker injection missed a
                        # canonical-collision case.
                        _bump_metric("vlm_retreat", request_id)
                        prompt_cache = None
                        rest_tokens = list(model_tokens)
                        cache_match_type = "miss"
                        matched_prefix_len = 0
                        cache_selection_source = "vlm_retreat"
                        _kv_off = None

                if SETTINGS.vlm_cache_debug and prompt_cache is not None:
                    _terminal_status(
                        "🔬",
                        f"[cache-diag] match={cache_match_type} mpl={matched_prefix_len} "
                        f"kv_off={_kv_off} model_len={len(model_tokens)} "
                        f"canon_len={len(prompt_tokens)} delta={len(model_tokens)-len(prompt_tokens)} "
                        f"rest={len(rest_tokens)} layer_type={type(prompt_cache[0]).__name__ if prompt_cache else 'None'}",
                        indent=1,
                    )

                # --- M3: Stable-prefix fallback ---
                # If the global cache lookup found fewer cached tokens than the
                # message-level diff says are stable, try a secondary lookup
                # using the stable prefix as the minimum acceptable match.
                # Only triggers when there is a real improvement available.
                if (
                    stable_prefix_token_len_computed > matched_prefix_len
                    and stable_prefix_token_len_computed > 0
                    and stable_prefix_token_len_computed < len(prompt_tokens)
                ):
                    # Try to find a cache entry that covers at least stable_prefix_token_len_computed tokens.
                    # We look up the stable prefix tokens directly (canonical key).
                    stable_prefix_toks = prompt_tokens[
                        :stable_prefix_token_len_computed
                    ]
                    # Also scope the stable-prefix lookup to the request's
                    # model_tokens so any candidate that shares the canonical
                    # prefix but differs in model space (e.g. different image
                    # expansion) is rejected instead of silently producing a
                    # bad KV reuse.
                    (
                        sp_cache,
                        sp_rest_model_tokens,
                        sp_cache_session_tokens,
                        sp_match_type,
                        sp_matched_prefix_len,
                    ) = PROMPT_CACHE.fetch_nearest_cache(
                        SETTINGS.model_path,
                        stable_prefix_toks,
                        request_model_tokens=model_tokens,
                    )
                    if (
                        sp_cache is not None
                        and sp_matched_prefix_len > matched_prefix_len
                    ):
                        # fetch_nearest_cache already trimmed sp_cache in
                        # model space via LCP(entry.model_tokens,
                        # request model_tokens), so sp_rest_model_tokens is
                        # the correct tail to prefill.  No further canonical
                        # trim math here — that was the VLM mis-slice bug.
                        rest_tokens = (
                            sp_rest_model_tokens
                            if sp_rest_model_tokens is not None
                            else list(model_tokens)
                        )
                        prompt_cache = sp_cache
                        cache_session_tokens = sp_cache_session_tokens
                        matched_prefix_len = sp_matched_prefix_len
                        cache_match_type = "stable_prefix_" + sp_match_type
                        cache_selection_source = "stable_prefix"

                        # Defensive VLM safety net (same rationale as the
                        # primary lookup path — the new fetch should never
                        # leave a partial image, but guard in case a future
                        # tokenizer change sneaks past LCP).
                        if is_vlm:
                            _is_partial, _full_pads, _rest_pads = (
                                _vlm_cache_covers_partial_image(
                                    model_tokens, rest_tokens
                                )
                            )
                            if _is_partial:
                                if SETTINGS.vlm_cache_debug:
                                    _terminal_status(
                                        "🛟",
                                        f"[vlm-cache-retreat] stable-prefix cut mid-image "
                                        f"(full_pads={_full_pads} rest_pads={_rest_pads}) — "
                                        "dropping cache and prefilling fresh.",
                                        indent=1,
                                    )
                                # 1a: evict the offending stable-prefix
                                # entry for the same reason as the primary
                                # retreat path — break the loop.
                                try:
                                    if cache_session_tokens:
                                        PROMPT_CACHE.delete(
                                            SETTINGS.model_path, cache_session_tokens
                                        )
                                except Exception:
                                    pass
                                # P4: same rationale as primary retreat.
                                _bump_metric("vlm_retreat", request_id)
                                prompt_cache = None
                                rest_tokens = list(model_tokens)
                                cache_match_type = "miss"
                                matched_prefix_len = 0
                                cache_selection_source = "vlm_retreat"

            # --- DEBUG BLOCK ---
            # Only fire when the cache divergence is genuinely unexpected:
            # - On a full miss (no cache entry found at all), or
            # - On a 'shorter' hit where the matched prefix is significantly
            #   less than what the stable-prefix layer expected to be cached.
            #   A 'shorter' hit at the normal end-of-last-turn boundary is
            #   expected behaviour and should not generate noise.
            _debug_unexpected_miss = cache_match_type == "miss" or (
                cache_match_type in ("shorter",)
                and stable_prefix_token_len_computed > 0
                and matched_prefix_len < stable_prefix_token_len_computed - 64
            )
            if (
                SETTINGS.vlm_cache_debug
                and cache_session_tokens
                and _debug_unexpected_miss
            ):
                # Compare the prompt against the actual candidate the global cache evaluated
                _debug_token_divergence(
                    tokenizer, prompt_tokens, cache_session_tokens, context_window=8
                )
            # -----------------------

            if prompt_cache is None:
                cache_model = (
                    model.language_model
                    if is_vlm and hasattr(model, "language_model")
                    else model
                )
                prompt_cache = make_prompt_cache(
                    cache_model, max_kv_size=SETTINGS.max_kv_size
                )
                # Full miss: model prefills all original tokens (never canonical).
                rest_tokens = model_tokens
            rest_count = (
                len(rest_tokens) if rest_tokens is not None else len(model_tokens)
            )
            cache_session_id = _cache_log_session_id(session_ctx, cache_session_tokens)
            if SETTINGS.enable_request_logging:
                try:
                    request_logger = CacheSessionTranscriptLogger(
                        cache_session_id=cache_session_id
                    )
                except Exception:
                    request_logger = None

            if request_logger:
                request_logger.log(
                    "prompt",
                    {
                        "request_meta": {
                            "path": self.path,
                            "stream": bool(body.get("stream", False)),
                            "model": body.get("model", SETTINGS.proxy_model_id),
                            "model_family": SETTINGS.model_family,
                            "temperature": body.get(
                                "temperature", SETTINGS.default_temperature
                            ),
                            "max_tokens": body.get(
                                "max_tokens", SETTINGS.default_max_tokens
                            ),
                            "enable_thinking": enable_thinking,
                            "thinking_source": reasoning_control["source"],
                            "thinking_raw": reasoning_control["raw"],
                            "cache_session_id": cache_session_id,
                            "cache_match_type": cache_match_type,
                            "cache_selection_source": cache_selection_source,
                            "matched_prefix_len": matched_prefix_len,
                            "cache_prompt_normalized": prompt_was_normalized,
                            "cache_key_normalized": prompt_was_normalized,
                            "cache_key_delta_chars": cache_key_delta_chars,
                            "prompt_tokens": len(prompt_tokens),
                            "session_id": session_ctx.session_id,
                            "parent_session_id": session_ctx.parent_session_id,
                            "branch_id": session_ctx.branch_id,
                            "session_source": session_ctx.source,
                            # --- M6 telemetry: stable-prefix metrics ---
                            "stable_prefix_msg_count": stable_prefix_msg_count_computed,
                            "stable_prefix_token_len": stable_prefix_token_len_computed,
                            "stable_prefix_diff": stable_prefix_diff_descriptors[
                                :8
                            ],  # cap to avoid log bloat
                            # --- P4: cache-correctness metrics ---
                            "cache_metrics_snapshot": _metrics_snapshot(),
                            "cache_metrics_bumped_this_request": _metrics_drain_request(
                                request_id
                            ),
                            **(
                                {
                                    "vlm_format_prefix_stable": getattr(
                                        _vlm_diagnostics, "used_prefix_stable", False
                                    ),
                                    "prompt_token_prefix": list(prompt_tokens[:64]),
                                }
                                if is_vlm
                                else {}
                            ),
                        },
                        "messages": messages,
                        "tools": tools,
                        "rendered_prompt": prompt,
                    },
                    request_id=request_id,
                )
                sampler_payload = {
                    "applied_kwargs": sampler_kwargs,
                    "rest_tokens": rest_count,
                    "matched_prefix_len": matched_prefix_len,
                }
                if SETTINGS.vlm_cache_debug and is_vlm and prompt_tokens:
                    sampler_payload["prompt_token_prefix"] = list(prompt_tokens[:64])
                request_logger.log("sampler", sampler_payload, request_id=request_id)

            prompt_len = len(prompt_tokens)
            hit_ratio = (
                (matched_prefix_len / prompt_len * 100) if prompt_len > 0 else 0.0
            )

            if hit_ratio >= 85:
                cache_light = "🟢"
            elif hit_ratio >= 50:
                cache_light = "🟡"
            else:
                cache_light = "🔴"

            _terminal_status(
                "📨",
                f"Request {request_id} | {cache_light} Cache: {hit_ratio:.1f}% ({cache_match_type}) | "
                f"tokens={matched_prefix_len}/{prompt_len} | rest={rest_count} | "
                f"stream={body.get('stream', False)} | thinking={enable_thinking}",
            )
            _terminal_status(
                "⚙️",
                f"Generation started | wait={wait_seconds:.2f}s | prefill={rest_count} | "
                f"session={session_ctx.session_id[:16]} ({cache_selection_source}) | family={SETTINGS.model_family}",
                indent=1,
            )

            if not is_streaming:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                generated_parts = []
                progress_last_at = time.time()
                for response in _stream_generate_unified(
                    rest_tokens,
                    max_tokens,
                    sampler,
                    prompt_cache,
                    vlm_pixel_values=vlm_pixel_values,
                    vlm_mask=vlm_mask,
                    vlm_kwargs=vlm_kwargs,
                    cache_match_type=cache_match_type,
                ):
                    generated_parts.append(response.text)
                    generated_tokens.append(int(response.token))
                    if first_token_at is None:
                        first_token_at = time.time()
                    if (
                        len(generated_tokens) % 64 == 0
                        and (time.time() - progress_last_at) >= 1.0
                    ):
                        progress_last_at = time.time()
                        _terminal_status(
                            "⏳",
                            f"Request {request_id} in progress | generated_tokens={len(generated_tokens)}",
                            indent=1,
                        )
                    # G2: cooperative abort between tokens.  If the watchdog
                    # fired we break now — partial KV state is NOT cached
                    # (see the gated _insert_cache_entries / session-turn /
                    # healing-store blocks below) so next-turn correctness
                    # is preserved; the client receives whatever generated
                    # so far with finish_reason="length".
                    if watchdog_flag.is_set():
                        generation_aborted = True
                        _terminal_status(
                            "⏱️",
                            f"Request {request_id} aborted by watchdog "
                            f"after {SETTINGS.generation_watchdog_seconds}s "
                            f"| partial_tokens={len(generated_tokens)}",
                            indent=1,
                        )
                        break
                response_text = "".join(generated_parts)
                raw_response_text = response_text
                response_text = _normalize_assistant_text(
                    response_text, enable_thinking, SETTINGS.model_family
                )
                message_text, tool_calls = _extract_openai_tool_calls(
                    response_text, SETTINGS.model_family
                )
                # Hide <think> blocks from the client whenever reasoning was requested.
                if enable_thinking:
                    message_text = _strip_thinking_from_content(message_text)

                    # --- HEALING STORE LOGIC ---
                    # Skip on abort: hashing partial output would let a
                    # client retry (same partial text → same hash) resurrect
                    # a truncated response from a different request.
                    if raw_response_text != message_text and not generation_aborted:
                        h = _get_healing_hash(message_text, tool_calls)
                        if h:
                            with HEALING_STORE_LOCK:
                                HEALING_STORE[h] = raw_response_text
                                HEALING_STORE.move_to_end(h, last=True)
                                while len(HEALING_STORE) > MAX_HEALING_STORE:
                                    HEALING_STORE.popitem(last=False)

                # G2: if aborted, force finish_reason=length (OpenAI standard
                # signal that output was truncated).  Tool-call extraction
                # on partial output is already benign — the regex either
                # finds a complete <tool_call>…</tool_call> or returns none.
                finish_reason = (
                    "length"
                    if generation_aborted
                    else ("tool_calls" if tool_calls else "stop")
                )
                cache_key.extend(generated_tokens)
                # G2: insert cache + advance session-turn record ONLY on
                # clean completion.  Partial KV state with a completion key
                # that ends mid-token would silently corrupt next-turn
                # attention; a half-written session-turn record would
                # misreport the stable-prefix boundary.
                if not generation_aborted:
                    with prompt_cache_lock:
                        _insert_cache_entries(
                            model_name=SETTINGS.model_path,
                            session_ctx=session_ctx,
                            cache_key=cache_key,
                            prompt_cache=prompt_cache,
                            generated_tokens=generated_tokens,
                            tool_calls=tool_calls,
                            prompt_model_tokens=model_tokens,
                        )
                        # --- M5: Update session turn record for next-turn stable-prefix lookup ---
                        if _session_id_for_turn:
                            _update_session_turn_store(
                                _session_id_for_turn,
                                messages,
                                prompt_tokens,
                            )

                response_id = f"chatcmpl-{int(time.time())}"
                # Additive reasoning extraction: compute alongside the legacy
                # `message_text` so clients that understand `reasoning_content`
                # (PI, OpenAI Responses-aware tools) render the thinking block,
                # and clients that don't see byte-identical content.
                _, reasoning_text_nonstream = (
                    _split_text_for_reasoning(
                        raw_response_text, enable_thinking, SETTINGS.model_family
                    )
                    if enable_thinking
                    else ("", "")
                )
                assistant_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message_text,
                }
                if reasoning_text_nonstream:
                    assistant_msg["reasoning_content"] = reasoning_text_nonstream
                full_response = {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": SETTINGS.proxy_model_id,
                    "choices": [
                        {
                            "index": 0,
                            "message": assistant_msg,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                }
                if tool_calls:
                    full_response["choices"][0]["message"]["tool_calls"] = tool_calls
                self.wfile.write(json.dumps(full_response).encode("utf-8"))
                if request_logger:
                    timing = {}
                    if first_token_at is not None and generation_started_at is not None:
                        timing["prefill_seconds"] = (
                            first_token_at - generation_started_at
                        )
                        timing["decode_seconds"] = time.time() - first_token_at
                        timing["prefill_tps"] = (
                            rest_count / timing["prefill_seconds"]
                            if timing["prefill_seconds"] > 0
                            else None
                        )
                        timing["decode_tps"] = (
                            len(generated_tokens) / timing["decode_seconds"]
                            if timing["decode_seconds"] > 0
                            else None
                        )
                    request_logger.log(
                        "generation",
                        {
                            "mode": "non-stream",
                            "timing": timing,
                            "raw_response_text": raw_response_text,
                            "normalized_response_text": response_text,
                            "assistant_message_text": message_text,
                            "tool_calls": tool_calls,
                            "finish_reason": finish_reason,
                        },
                        request_id=request_id,
                    )

            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                # Close the connection after the stream completes — the
                # single-threaded HTTPServer otherwise parks on this socket
                # waiting for a follow-up request and starves new connections.
                # (_mark_connection_close() already set close_connection=True;
                # this header just tells the peer the same thing.)
                self.send_header("Connection", "close")
                self.end_headers()

                response_id = f"chatcmpl-{int(time.time())}"
                role_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": SETTINGS.proxy_model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()

                raw_parts = []
                message_text_parts: List[str] = []
                reasoning_text_parts: List[str] = []
                progress_last_at = time.time()
                splitter = _ReasoningStreamSplitter(
                    enable_thinking, SETTINGS.model_family
                )

                def _emit_delta(kind: str, text: str) -> None:
                    """Send an SSE chunk with reasoning_content or content delta."""
                    if not text:
                        return
                    field = (
                        "reasoning_content" if kind == "reasoning" else "content"
                    )
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": SETTINGS.proxy_model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {field: text},
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(
                        f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()

                for response in _stream_generate_unified(
                    rest_tokens,
                    max_tokens,
                    sampler,
                    prompt_cache,
                    vlm_pixel_values=vlm_pixel_values,
                    vlm_mask=vlm_mask,
                    vlm_kwargs=vlm_kwargs,
                    cache_match_type=cache_match_type,
                ):
                    generated_tokens.append(int(response.token))
                    if first_token_at is None:
                        first_token_at = time.time()
                    response_text = response.text
                    if response_text:
                        raw_parts.append(response_text)
                        # Live split + emit.  Tool-call text is accumulated in
                        # the splitter's tool buffer and NOT emitted here — it
                        # will surface at flush() for `_extract_openai_tool_calls`
                        # so we can emit `delta.tool_calls` deltas cleanly.
                        for kind, chunk_text in splitter.feed(response_text):
                            if kind == "reasoning":
                                reasoning_text_parts.append(chunk_text)
                            else:
                                message_text_parts.append(chunk_text)
                            _emit_delta(kind, chunk_text)
                    if (
                        len(generated_tokens) % 64 == 0
                        and (time.time() - progress_last_at) >= 1.0
                    ):
                        progress_last_at = time.time()
                        _terminal_status(
                            "⏳",
                            f"Request {request_id} in progress | generated_tokens={len(generated_tokens)}",
                            indent=1,
                        )
                    # G2: cooperative abort between tokens — same rationale
                    # as the non-streaming path.  Partial KV state must not
                    # be cached; the streaming epilogue below emits a
                    # terminal chunk with finish_reason="length" so the
                    # client receives clean framing.
                    if watchdog_flag.is_set():
                        generation_aborted = True
                        _terminal_status(
                            "⏱️",
                            f"Request {request_id} aborted by watchdog "
                            f"after {SETTINGS.generation_watchdog_seconds}s "
                            f"| partial_tokens={len(generated_tokens)}",
                            indent=1,
                        )
                        break

                # Drain the splitter: emits residual content/reasoning chunks
                # and returns any buffered tool-call XML.
                tail_chunks, tool_buffer_text = splitter.flush()
                for kind, chunk_text in tail_chunks:
                    if kind == "reasoning":
                        reasoning_text_parts.append(chunk_text)
                    else:
                        message_text_parts.append(chunk_text)
                    _emit_delta(kind, chunk_text)

                raw_full_text = "".join(raw_parts)
                # Run the legacy end-of-stream pipeline on the FULL raw text
                # so cache insert / healing / tool-call extraction stay
                # byte-identical to the non-streaming path.  This preserves
                # the dual-pipeline invariant — live deltas are a view, not
                # a replacement for the canonical post-processing.
                full_text = _normalize_assistant_text(
                    raw_full_text, enable_thinking, SETTINGS.model_family
                )
                canonical_message_text, tool_calls = _extract_openai_tool_calls(
                    full_text, SETTINGS.model_family
                )
                if enable_thinking:
                    canonical_message_text = _strip_thinking_from_content(
                        canonical_message_text
                    )

                # When the splitter entered TOOL mode we held back all text
                # from the first `<tool_call>` sentinel onward.  Re-running
                # `_extract_openai_tool_calls` on that buffer yields any
                # surrounding-but-non-tool-call text; emit it now as a final
                # content delta so the client sees text before AND after tool
                # calls (mirroring non-streaming `message.content`).
                if tool_buffer_text:
                    trailing_content, _ = _extract_openai_tool_calls(
                        tool_buffer_text, SETTINGS.model_family
                    )
                    trailing_content = trailing_content.strip()
                    if trailing_content:
                        message_text_parts.append(trailing_content)
                        _emit_delta("content", trailing_content)

                # Healing hash uses the stripped text — what the client will
                # echo back in the next turn's assistant message — so the
                # canonical message_text is the correct key.  The raw stream
                # text is what we heal back to on next-turn replay.
                if enable_thinking:
                    if (
                        raw_full_text != canonical_message_text
                        and not generation_aborted
                    ):
                        h = _get_healing_hash(canonical_message_text, tool_calls)
                        if h:
                            with HEALING_STORE_LOCK:
                                HEALING_STORE[h] = raw_full_text
                                HEALING_STORE.move_to_end(h, last=True)
                                while len(HEALING_STORE) > MAX_HEALING_STORE:
                                    HEALING_STORE.popitem(last=False)

                # Alias names used downstream for logging + cache insert.
                message_text = canonical_message_text
                full_text = canonical_message_text

                cache_key.extend(generated_tokens)
                # G2: only insert + advance session on clean completion.
                if not generation_aborted:
                    with prompt_cache_lock:
                        _insert_cache_entries(
                            model_name=SETTINGS.model_path,
                            session_ctx=session_ctx,
                            cache_key=cache_key,
                            prompt_cache=prompt_cache,
                            generated_tokens=generated_tokens,
                            tool_calls=tool_calls,
                            prompt_model_tokens=model_tokens,
                        )
                        # --- M5: Update session turn record for next-turn stable-prefix lookup ---
                        if _session_id_for_turn:
                            _update_session_turn_store(
                                _session_id_for_turn,
                                messages,
                                prompt_tokens,
                            )

                if tool_calls:
                    for idx, tc in enumerate(tool_calls):
                        tc_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": SETTINGS.proxy_model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"tool_calls": [{**tc, "index": idx}]},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        self.wfile.write(
                            f"data: {json.dumps(tc_chunk)}\n\n".encode("utf-8")
                        )
                        self.wfile.flush()

                # G2: override finish_reason on abort — OpenAI-standard
                # "length" tells the client the response was truncated.
                finish_reason = (
                    "length"
                    if generation_aborted
                    else ("tool_calls" if tool_calls else "stop")
                )
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": SETTINGS.proxy_model_id,
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": finish_reason}
                    ],
                }
                self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                if request_logger:
                    timing = {}
                    if first_token_at is not None and generation_started_at is not None:
                        timing["prefill_seconds"] = (
                            first_token_at - generation_started_at
                        )
                        timing["decode_seconds"] = time.time() - first_token_at
                        timing["prefill_tps"] = (
                            rest_count / timing["prefill_seconds"]
                            if timing["prefill_seconds"] > 0
                            else None
                        )
                        timing["decode_tps"] = (
                            len(generated_tokens) / timing["decode_seconds"]
                            if timing["decode_seconds"] > 0
                            else None
                        )
                    request_logger.log(
                        "generation",
                        {
                            "mode": "stream",
                            "timing": timing,
                            "raw_response_text": raw_full_text,
                            "normalized_response_text": full_text,
                            "assistant_message_text": message_text,
                            "tool_calls": tool_calls,
                            "finish_reason": finish_reason,
                        },
                        request_id=request_id,
                    )

        except BrokenPipeError:
            _terminal_status(
                "⚠️",
                f"Request {request_id} client disconnected (BrokenPipeError)",
                indent=1,
            )
            if request_logger:
                request_logger.log(
                    "generation",
                    "client disconnected (BrokenPipeError)",
                    request_id=request_id,
                )
        except Exception as e:
            _terminal_status("❌", f"Request {request_id} failed: {e}", indent=1)
            if request_logger:
                request_logger.log("generation", f"error: {e}", request_id=request_id)
        finally:
            if generation_started_at is not None:
                end_at = time.time()
                elapsed = max(end_at - generation_started_at, 1e-9)
                output_tokens = len(generated_tokens)
                speed = output_tokens / elapsed if output_tokens else 0.0

                non_reasoning_tokens = (
                    len(_tokenize_prompt(message_text)) if message_text else 0
                )
                reasoning_tokens = max(0, output_tokens - non_reasoning_tokens)
                token_breakdown = (
                    f"{output_tokens} (reasoning: {reasoning_tokens}, output: {non_reasoning_tokens})"
                    if enable_thinking
                    else f"{output_tokens}"
                )

                if first_token_at is not None:
                    prefill_seconds = first_token_at - generation_started_at
                    decode_seconds = max(end_at - first_token_at, 1e-9)
                    decode_tps = (
                        output_tokens / decode_seconds if output_tokens else 0.0
                    )
                    prefill_tps = (
                        rest_count / prefill_seconds if prefill_seconds > 0 else 0.0
                    )
                    _terminal_status(
                        "✅",
                        (
                            f"Request {request_id} finished | output_tokens={token_breakdown} | "
                            f"elapsed={elapsed:.2f}s | tok/s={speed:.2f} | "
                            f"prefill={prefill_seconds:.2f}s ({prefill_tps:.0f} tok/s) | "
                            f"decode={decode_seconds:.2f}s ({decode_tps:.1f} tok/s)"
                        ),
                        indent=1,
                    )
                else:
                    _terminal_status(
                        "✅",
                        (
                            f"Request {request_id} finished | output_tokens={token_breakdown} | "
                            f"elapsed={elapsed:.2f}s | tok/s={speed:.2f}"
                        ),
                        indent=1,
                    )
            # G2: cancel the watchdog timer unconditionally.  cancel() is
            # idempotent on Timer — safe if it already fired.  Must happen
            # BEFORE releasing model_lock so a slow cancel can't race with
            # the next request acquiring the lock and seeing a leftover
            # flag from a prior request (the flag is per-request-local so
            # this is extra belt-and-braces, but cheap).
            if watchdog_timer is not None:
                try:
                    watchdog_timer.cancel()
                except Exception:
                    pass
            # P4: drop any unconsumed per-request bump records (non-stream
            # path consumes via _metrics_drain_request; log failures or
            # early exits would otherwise leak this dict entry).  Also
            # clear the thread-local request id so later bump sites on the
            # same thread don't misattribute to a finished request.
            try:
                _metrics_drain_request(request_id)
            except Exception:
                pass
            _set_current_request_id(None)
            # On normal exit or Python exception, release the lock. On process abort (e.g. Metal
            # "uncommitted encoder" crash), finally may not run, so the "leaked semaphore" warning
            # at shutdown is expected; fixing the Metal crash resolves it.
            if acquired:
                model_lock.release()


def run():
    start_litellm_proxy()
    server_address = (SETTINGS.mlx_host, SETTINGS.mlx_port)
    # MLX 0.31 streams are strictly per-thread and mlx-vlm 0.4.4's generate_step
    # issues bare `mx.async_eval(y)` calls outside a `with mx.stream(...)` block,
    # which crashes in worker threads ("There is no Stream(gpu, N) in current
    # thread."). Since all GPU work already serializes through model_lock, we
    # serve requests single-threaded from the main thread where the module-level
    # `generation_stream` was created. When upstream mlx-vlm moves to
    # `mx.new_thread_local_stream` (or moves the async_eval inside a stream
    # context) this can revert to ThreadingHTTPServer.
    httpd = HTTPServer(server_address, APIHandler)

    print("\n" + "=" * 50)
    print("🟢 SYSTEM READY")
    print(f"   • Mode:         {'VLM (vision)' if is_vlm else 'LM (text-only)'}")
    print(f"   • MLX Engine:   http://{SETTINGS.mlx_host}:{SETTINGS.mlx_port}")
    print(f"   • LiteLLM:      http://127.0.0.1:{SETTINGS.proxy_port}")
    print("=" * 50 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    run()
