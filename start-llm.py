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
from datetime import datetime
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache

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
    load_dotenv(dotenv_path=DOTENV_PATH)


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


def _env_kv_bits(name: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().upper()
    if normalized == "OFF":
        return None
    try:
        return int(normalized)
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
    kv_bits: Optional[int]
    default_temperature: float
    default_top_p: float
    default_top_k: int
    default_min_p: float
    default_repetition_penalty: float
    default_repetition_context_size: int
    default_max_tokens: int
    enable_request_logging: bool
    vlm_cache_debug: bool
    normalize_write_tool_content_for_prompt: bool
    cache_canonicalize_tool_context: bool
    cache_session_partitioning: bool
    prompt_cache_block_size: int
    cache_use_block_index: bool
    log_root: Path
    proxy_startup_wait_seconds: float
    proxy_model_id: str


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
        default_temperature=_env_float("DEFAULT_TEMPERATURE", 0.1),
        default_top_p=_env_float("DEFAULT_TOP_P", 0.9),
        default_top_k=_env_int("DEFAULT_TOP_K", 40),
        default_min_p=_env_float("DEFAULT_MIN_P", 0.05),
        default_repetition_penalty=_env_float("DEFAULT_REPETITION_PENALTY", 1.08),
        default_repetition_context_size=_env_int("DEFAULT_REPETITION_CONTEXT_SIZE", 256),
        default_max_tokens=_env_int("DEFAULT_MAX_TOKENS", 2048),
        enable_request_logging=_env_bool("ENABLE_REQUEST_LOGGING", True),
        vlm_cache_debug=_env_bool("VLM_CACHE_DEBUG", False),
        normalize_write_tool_content_for_prompt=_env_bool("NORMALIZE_WRITE_TOOL_CONTENT_FOR_PROMPT", False),
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
        log_root=Path(_env_str("LOG_ROOT", str(SCRIPT_DIR / "logs"))),
        proxy_startup_wait_seconds=_env_float("PROXY_STARTUP_WAIT_SECONDS", 2.0),
        proxy_model_id=proxy_model_id,
    )


SETTINGS = _build_settings()

# VLM model_type whitelist (from mlx-vlm prompt_utils / supported architectures).
VLM_MODEL_TYPES = frozenset({
    "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe",
    "qwen3_5", "qwen3_5_moe", "qwen3_omni_moe",
    "glm4v", "glm4v_moe", "glm_ocr",
    "llava", "llava_next", "llava_qwen2", "llava_bunny",
    "idefics2", "idefics3", "mistral3",
    "gemma3", "gemma3n", "pixtral",
    "deepseek_vl_v2", "deepseekocr", "deepseekocr_2",
    "aya_vision", "cohere2_vision", "internvl_chat",
    "kimi_vl", "molmo", "molmo2", "smolvlm",
    "jina_vlm", "jvlm", "phi3_v", "paligemma",
    "florence2", "multi_modality", "mllama", "llama4",
    "dots_ocr", "paddleocr_vl", "ernie4_5_moe_vl",
    "lfm2_vl", "hunyuan_vl", "bunny-llama",
})


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
            path = Path(snapshot_download(repo_id=path_str, allow_patterns=["*.json", "*.safetensors", "*.model", "*.tiktoken", "*.py", "*.jinja"]))
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

TOOL_CALL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
ARG_PAIR_PATTERN = re.compile(
    r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
    re.DOTALL | re.IGNORECASE,
)
QWEN_FUNCTION_PATTERN = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.DOTALL | re.IGNORECASE)
QWEN_PARAMETER_PATTERN = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL | re.IGNORECASE)
INBOUND_META_MESSAGE_ID_PATTERN = re.compile(
    r'("message_id"\s*:\s*")[^"]+(")'
)
INBOUND_TRUSTED_CONTEXT_BLOCK_PATTERN = re.compile(
    r"## Group Chat Context\s*\n"
    r"## Inbound Context \(trusted metadata\)\s*\n"
    r".*?```json\s*\n.*?\n```\s*\n?",
    re.DOTALL,
)
CACHE_STABLE_INBOUND_CONTEXT_BLOCK = (
    "## Group Chat Context\n"
    "## Inbound Context (trusted metadata)\n"
    "__CACHE_STABLE_INBOUND_CONTEXT__\n"
)
INBOUND_CONTEXT_TO_PROJECT_BOUNDARY_PATTERN = re.compile(
    r"(__CACHE_STABLE_INBOUND_CONTEXT__)\n+# Project Context"
)
# Strip reasoning from response content when returning to client (so reasoning is hidden).
# Full think blocks (<think>...</think>) and "orphan" </think> (reasoning with no opening tag, e.g. GLM-style).
THINK_TAG_STRIP_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
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


# Session store of full assistant content (with <think>) for VLM cache-key stability.
# Key (model_path, session_id) -> last full assistant text. Client sends stripped content;
# we substitute stored full content when building the next prompt so cache keys match.
SESSION_FULL_ASSISTANT: OrderedDict = OrderedDict()
SESSION_FULL_ASSISTANT_LOCK = threading.Lock()
MAX_SESSION_FULL_ASSISTANT = 500

# Token trajectory store: exact token IDs per assistant turn for cache-key stitching.
# Key (model_path, stable_session_key) -> list of token ID lists, one per assistant turn.
# When building the next request we stitch tokenize(prefix) + stored_turn_ids + tokenize(suffix)
# so the token sequence matches the stored KV cache (no re-tokenization divergence).
TOKEN_TRAJECTORY_STORE: OrderedDict = OrderedDict()
TOKEN_TRAJECTORY_STORE_LOCK = threading.Lock()
MAX_TRAJECTORY_TURNS_PER_CONVERSATION = 64
MAX_TRAJECTORY_CONVERSATIONS = 500


def _token_trajectory_append(key: Tuple[str, str], generated_tokens: List[int]) -> None:
    """Append one assistant turn's token IDs to the trajectory store; evict if over cap."""
    with TOKEN_TRAJECTORY_STORE_LOCK:
        turns = TOKEN_TRAJECTORY_STORE.get(key, [])
        turns = list(turns) if isinstance(turns, list) else []
        turns.append(list(generated_tokens))
        if len(turns) > MAX_TRAJECTORY_TURNS_PER_CONVERSATION:
            turns = turns[-MAX_TRAJECTORY_TURNS_PER_CONVERSATION:]
        TOKEN_TRAJECTORY_STORE[key] = turns
        TOKEN_TRAJECTORY_STORE.move_to_end(key, last=True)
        while len(TOKEN_TRAJECTORY_STORE) > MAX_TRAJECTORY_CONVERSATIONS:
            TOKEN_TRAJECTORY_STORE.popitem(last=False)


def _terminal_status(icon: str, message: str, indent: int = 0) -> None:
    with console_lock:
        ts = datetime.now().strftime("%H:%M:%S")
        pad = "  " * max(indent, 0)
        print(f"{pad}{icon} [{ts}] {message}", flush=True)

def _debug_token_divergence(tokenizer, current_tokens: List[int], stored_tokens: Tuple[int, ...], context_window: int = 5):
    """Finds and prints exactly where two token sequences diverge for cache debugging."""
    min_len = min(len(current_tokens), len(stored_tokens))
    diverge_idx = -1
    
    for i in range(min_len):
        if current_tokens[i] != stored_tokens[i]:
            diverge_idx = i
            break
            
    if diverge_idx == -1:
        if len(current_tokens) == len(stored_tokens):
            _terminal_status("🔍", "DEBUG: Token sequences are completely identical!")
        else:
            _terminal_status("🔍", f"DEBUG: Perfect prefix match! Lengths: {len(current_tokens)} vs {len(stored_tokens)}")
        return

    # Calculate window bounds for context
    start_idx = max(0, diverge_idx - context_window)
    end_idx_current = min(len(current_tokens), diverge_idx + context_window + 1)
    end_idx_stored = min(len(stored_tokens), diverge_idx + context_window + 1)

    print(f"\n" + "="*50)
    print(f"🚨 CACHE DIVERGENCE DETECTED AT INDEX {diverge_idx} 🚨")
    print(f"Token IDs before divergence: {current_tokens[start_idx:diverge_idx]}")
    
    # Try to decode the text for human readability
    try:
        matching_text = tokenizer.decode(current_tokens[start_idx:diverge_idx])
        print(f"Matching text leading up: {repr(matching_text)}")
        
        curr_divergent_token = current_tokens[diverge_idx]
        stor_divergent_token = stored_tokens[diverge_idx]
        print(f"\n❌ Current Request Token [{diverge_idx}]: ID {curr_divergent_token} -> {repr(tokenizer.decode([curr_divergent_token]))}")
        print(f"❌ Stored Cache Token  [{diverge_idx}]: ID {stor_divergent_token} -> {repr(tokenizer.decode([stor_divergent_token]))}")
        
        curr_context_after = tokenizer.decode(current_tokens[diverge_idx+1:end_idx_current])
        stor_context_after = tokenizer.decode(stored_tokens[diverge_idx+1:end_idx_stored])
        print(f"\nCurrent context after: {repr(curr_context_after)}")
        print(f"Stored context after:  {repr(stor_context_after)}")
    except Exception as e:
        print(f"Could not decode tokens: {e}")
    print("="*50 + "\n")


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
        tokens: Tuple[int, ...]
        count: int
        touched_at: float

    def __init__(self, max_size=10, ttl_seconds=1800):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        # Force block size to a minimum of 16, overriding the disabled config
        self.block_size = max(16, getattr(SETTINGS, "prompt_cache_block_size", 16))
        
        # 1. The Core Flat Cache: (model, exact_tokens) -> CacheEntry
        self._entries: Dict[Tuple[str, Tuple[int, ...]], self.CacheEntry] = {}
        
        # 2. The Block Hash Index: (model, chain_hash) -> Set of token sequences containing this block
        self._block_index: Dict[Tuple[str, bytes], set] = {}
        
        self._lru = deque()

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
        try:
            self._lru.remove(key)
        except ValueError:
            pass

    def _extract(self, model, tokens):
        key = (model, tuple(tokens))
        entry = self._entries[key]
        entry.touched_at = time.time()
        
        # Never delete the cache on read. Multi-agent branching requires 
        # the root prefix to remain alive in VRAM for other agents to share.
        # The LRU pruner will safely handle VRAM limits.
        
        try:
            self._lru.remove(key)
        except ValueError:
            pass
        self._lru.append(key)
        
        return self.CacheEntry(copy.deepcopy(entry.prompt_cache), entry.tokens, entry.count, time.time())

    def fetch_nearest_cache(self, model, tokens):
        self.prune_expired()
        tokens_tup = tuple(tokens)
        
        # 1. Hash the incoming request into blocks
        chain_pairs = _block_chain_hashes(tokens_tup, self.block_size)
        if not chain_pairs:
            return None, tokens, tokens, "miss", 0

        best_prefix_len = 0
        best_cached_tokens = None
        
        # 2. Walk the blocks BACKWARDS to find the longest matching chain hash instantly
        for chain_hash, req_prefix_len in reversed(chain_pairs):
            idx_key = (model, chain_hash)
            if idx_key in self._block_index:
                # We found active cache entries containing this exact block prefix
                candidate_tokens_set = self._block_index[idx_key]
                for candidate_tokens in candidate_tokens_set:
                    # Double-check exact token match up to prefix_len to avoid SHA256 collisions
                    if tokens_tup[:req_prefix_len] == candidate_tokens[:req_prefix_len]:
                        best_prefix_len = req_prefix_len
                        best_cached_tokens = candidate_tokens
                        break
            
            if best_cached_tokens is not None:
                break # Found the longest block match

        # 3. If no block matched, return miss
        if best_cached_tokens is None:
            return None, tokens, tokens, "miss", 0

        # 4. We found a match. Extract the cache entry.
        entry = self._extract(model, best_cached_tokens)

        # 5. Check if remaining tokens within the current block also match
        while best_prefix_len < min(len(tokens_tup), len(best_cached_tokens)):
            if tokens_tup[best_prefix_len] == best_cached_tokens[best_prefix_len]:
                best_prefix_len += 1
            else:
                break

        # 6. Return the sliced/trimmed cache
        if best_prefix_len == len(tokens_tup):
            # Exact Match
            if len(tokens_tup) > 1 and can_trim_prompt_cache(entry.prompt_cache):
                trim_prompt_cache(entry.prompt_cache, 1)
                return entry.prompt_cache, tokens_tup[-1:], best_cached_tokens, "exact", len(tokens_tup) - 1
            return entry.prompt_cache, tokens_tup, best_cached_tokens, "exact", len(tokens_tup)
            
        elif best_prefix_len < len(tokens_tup):
            # Shorter Match (We have the prefix, compute the rest)
            if can_trim_prompt_cache(entry.prompt_cache):
                trim_prompt_cache(entry.prompt_cache, len(best_cached_tokens) - best_prefix_len)
            return entry.prompt_cache, list(tokens_tup)[best_prefix_len:], best_cached_tokens, "shorter", best_prefix_len

        elif best_prefix_len > len(tokens_tup):
            # Longer cache: request is a prefix of cached; trim cache to request length
            if can_trim_prompt_cache(entry.prompt_cache):
                num_to_trim = len(best_cached_tokens) - len(tokens_tup)
                trim_prompt_cache(entry.prompt_cache, num_to_trim)
                return entry.prompt_cache, [], best_cached_tokens, "longer", len(tokens_tup)
        return None, tokens, tokens, "miss", 0

    def insert_cache(self, model, tokens, prompt_cache):
        self.prune_expired()
        tokens_tup = tuple(tokens)
        key = (model, tokens_tup)
        
        if key in self._entries:
            self._entries[key].count += 1
            self._entries[key].touched_at = time.time()
            self._lru.remove(key)
            self._lru.append(key)
            return

        # Insert into flat dictionary
        self._entries[key] = self.CacheEntry(prompt_cache, tokens_tup, 1, time.time())
        
        # Map every block hash to this token sequence
        chain_pairs = _block_chain_hashes(tokens_tup, self.block_size)
        for chain_hash, _ in chain_pairs:
            idx_key = (model, chain_hash)
            if idx_key not in self._block_index:
                self._block_index[idx_key] = set()
            self._block_index[idx_key].add(tokens_tup)

        self._lru.append(key)
        
        if len(self._lru) > self.max_size:
            old_key = self._lru.popleft()
            self._delete(old_key[0], old_key[1])

    def contains_tokens(self, model, tokens):
        key = (model, tuple(tokens))
        return key in self._entries

    def extract_exact_cache(self, model, tokens):
        if not self.contains_tokens(model, tokens):
            return None
        return self._extract(model, tuple(tokens))

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
    def _append_unique_bounded(queue: deque, key_tuple: Tuple[int, ...], limit: int) -> None:
        try:
            queue.remove(key_tuple)
        except ValueError:
            pass
        queue.append(key_tuple)
        while len(queue) > limit:
            queue.popleft()

    def register_cache_key(self, session_ctx: SessionContext, cache_key: List[int]) -> None:
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

        if session_ctx.parent_session_id and session_ctx.parent_session_id != session_id:
            state.parent_session_id = session_ctx.parent_session_id
        state.touched_at = time.time()

        key_tuple = tuple(cache_key)
        self._append_unique_bounded(state.keys, key_tuple, self._max_entries_per_session)

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
            self._append_unique_bounded(state.anchors, key_tuple, self._max_anchor_entries)

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
            if len(prompt_tokens) > 1 and can_trim_prompt_cache(cache_entry.prompt_cache):
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
    ):
        """Always use global cache; session/conversation/thread ID is ignored for lookup."""
        prompt_cache_store.prune_expired()
        self._prune_idle()
        selected = prompt_cache_store.fetch_nearest_cache(model_name, prompt_tokens)
        return (*selected, "global")


SESSION_INDEX = SessionIndex(
    max_entries_per_session=SETTINGS.prompt_cache_max_entries_per_session,
    max_idle_seconds=SETTINGS.prompt_cache_session_max_idle_seconds,
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
        self._write_line(f"[{self._ts()}] request_id={request_id} direction={direction}")
        if isinstance(payload, (dict, list)):
            self._write_line(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            self._write_line(str(payload))
        self._write_line("")


def _cache_session_id(tokens: List[int]) -> str:
    raw = ",".join(str(tok) for tok in tokens[:1024])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_log_session_id(session_ctx: SessionContext, cache_session_tokens: List[int]) -> str:
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


def _vlm_stable_session_key(messages: List[Dict[str, Any]]) -> str:
    """
    Derive a stable session key from the first two messages (system + first user)
    so that SESSION_FULL_ASSISTANT lookup/store matches across turns. Used when
    session_id is implicit (derived from prompt_tokens), because for VLM we don't
    have prompt_tokens yet when we need to look up the stored full assistant.
    """
    parts = []
    for msg in messages[:2]:
        role = (msg.get("role") or "").strip()
        content = _flatten_content(msg.get("content", ""))
        parts.append(f"{role}:{content[:4096]}")
    raw = "\n".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _vlm_assistant_block_wrapper(
    processor_any: Any,
    enable_thinking: bool,
    model_family: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Return (prefix, suffix) strings that wrap assistant content in the chat template.
    Used for token trajectory stitching: we tokenize prefix + stored_turn_ids + suffix
    so we never re-tokenize assistant content. Template-aware for Qwen; generic fallback.
    """
    family = (model_family or getattr(SETTINGS, "model_family", "") or "").strip().lower()
    if family == "qwen3" or "qwen" in family:
        # Qwen: <|im_start|>assistant\n then optional <think>\n then content then \n</think>\n<|im_end|>
        prefix = "<|im_start|>assistant\n"
        if enable_thinking:
            prefix += "<think>\n"
        suffix = "\n</think>\n<|im_end|>\n" if enable_thinking else "\n<|im_end|>\n"
        return prefix, suffix
    if family == "glm4":
        # GLM4 VLM: similar structure; use generic if needed.
        prefix = "<|assistant|>\n"
        suffix = "\n"
        return prefix, suffix
    # Generic: minimal wrapper so tokenizer does not merge across boundaries.
    prefix = "<|im_start|>assistant\n"
    suffix = "\n<|im_end|>\n"
    return prefix, suffix


def _vlm_build_stitched_prompt_tokens(
    processor_any: Any,
    messages: List[Dict[str, Any]],
    trajectory_turns: List[List[int]],
    tools: Optional[Any],
    enable_thinking: Optional[bool],
    model_family: Optional[str] = None,
) -> List[int]:
    """
    Build prompt token list by stitching tokenized segments with exact stored assistant
    token IDs from TokenTrajectoryStore. Guarantees the token sequence matches the
    stored KV cache from previous turns (no re-tokenization divergence).
    """
    template_processor = None
    if processor_any is not None and hasattr(processor_any, "apply_chat_template"):
        if getattr(processor_any, "chat_template", None) is not None:
            template_processor = processor_any
    if template_processor is None and getattr(processor_any, "tokenizer", None) is not None:
        tok = processor_any.tokenizer
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None) is not None:
            template_processor = tok
    if template_processor is None or not messages:
        return []

    tokenizer = getattr(processor_any, "tokenizer", processor_any)
    template_kwargs: Dict[str, Any] = {}
    if tools is not None:
        template_kwargs["tools"] = tools
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking

    def _apply(mes: List[Dict[str, Any]], gen: bool) -> str:
        try:
            raw = template_processor.apply_chat_template(
                mes,
                tokenize=False,
                add_generation_prompt=gen,
                **template_kwargs,
            )
        except Exception:
            return ""
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        return str(raw) if raw else ""

    # Per-message segment strings (same logic as _vlm_format_prompt_prefix_stable).
    parts: List[str] = []
    try:
        prev = _apply([], False)
    except Exception:
        prev = ""
    for i in range(len(messages)):
        s = _apply(messages[: i + 1], False)
        if len(s) >= len(prev):
            parts.append(s[len(prev) :])
        else:
            parts.append("")
        prev = s
    full_no_gen = prev
    full_with_gen = _apply(messages, True)
    if full_with_gen.startswith(full_no_gen) and len(full_with_gen) >= len(full_no_gen):
        gen_suffix = full_with_gen[len(full_no_gen) :]
    else:
        gen_suffix = "\nAssistant:"

    asst_prefix, asst_suffix = _vlm_assistant_block_wrapper(
        processor_any, bool(enable_thinking), model_family
    )

    def _tokenize_segment(seg: str, add_special_tokens: bool) -> List[int]:
        if not seg:
            return []
        if tokenizer.bos_token is None or not add_special_tokens:
            return tokenizer.encode(seg, add_special_tokens=add_special_tokens)
        return tokenizer.encode(seg, add_special_tokens=add_special_tokens)

    out: List[int] = []
    assistant_count = sum(1 for msg in messages if (msg.get("role") or "").strip().lower() == "assistant")
    use_count = min(len(trajectory_turns), assistant_count)
    trajectory_start_idx = assistant_count - use_count

    assistant_idx = 0
    first = True
    for i, msg in enumerate(messages):
        seg = parts[i] if i < len(parts) else ""
        role = (msg.get("role") or "").strip().lower()
        if role == "assistant":
            if assistant_idx < trajectory_start_idx:
                # Fallback: tokenize segment as-is (no stored trajectory).
                out.extend(_tokenize_segment(seg, add_special_tokens=first))
                first = False
            else:
                out.extend(_tokenize_segment(asst_prefix, add_special_tokens=first))
                first = False
                turn_idx = assistant_idx - trajectory_start_idx
                out.extend(trajectory_turns[turn_idx])
                out.extend(_tokenize_segment(asst_suffix, add_special_tokens=False))
            assistant_idx += 1
        else:
            out.extend(_tokenize_segment(seg, add_special_tokens=first))
            first = False
    out.extend(_tokenize_segment(gen_suffix, add_special_tokens=False))
    return out


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
                            serialized = json.dumps(raw_content, ensure_ascii=False, sort_keys=True)
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
                url = u.get("url") if isinstance(u, dict) else u if isinstance(u, str) else None
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


def _prepare_messages_for_vlm(messages: List[Dict[str, Any]], tools: Optional[Any] = None) -> List[Dict[str, Any]]:
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
                    text = (part.get("text") or part.get("content") or "")
                    if SETTINGS.cache_canonicalize_tool_context:
                        text = _normalize_prompt_for_cache(text)
                    new_content.append({**part, "text": text})
                else:
                    new_content.append(part)
            m["content"] = new_content
        elif isinstance(content, str):
            if SETTINGS.cache_canonicalize_tool_context:
                m["content"] = _normalize_prompt_for_cache(content)
            else:
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


def _vlm_normalize_thinking_newlines(formatted: str) -> str:
    """
    Best-effort canonicalize newlines after <think> and </think> for tokenization stability.
    The canonical fix for cache-key divergence is Token Trajectory Stitching (stored
    exact token IDs per assistant turn); this normalization is fallback when trajectory
    is not used (e.g. first turn or eviction).
    """
    if not formatted:
        return formatted
    # After opening <think> or closing </think>, collapse any run of whitespace+newlines to single \\n
    formatted = re.sub(r"(<think>)\s*\n+", r"\1\n", formatted)
    formatted = re.sub(r"(</think>)\s*\n+", r"\1\n", formatted)
    return formatted


def _vlm_format_prompt_prefix_stable(
    processor_any,
    messages: List[Dict[str, Any]],
    add_generation_prompt: bool = True,
    **kwargs: Any,
) -> str:
    """
    Build the chat-formatted prompt so that the prefix for the first K messages
    is always the same token sequence regardless of how many messages follow.
    This enables prompt-cache prefix reuse across turns (cache=shorter).
    Falls back to get_chat_template if the processor has no apply_chat_template.
    Pass tools= and enable_thinking= (and any other template kwargs) so output
    matches the full template and cache prefixes align.
    """
    try:
        _vlm_diagnostics.used_prefix_stable = False
    except AttributeError:
        pass
    template_processor = None
    if processor_any is not None and hasattr(processor_any, "apply_chat_template"):
        if getattr(processor_any, "chat_template", None) is not None:
            template_processor = processor_any
    if template_processor is None and getattr(processor_any, "tokenizer", None) is not None:
        tok = processor_any.tokenizer
        if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None) is not None:
            template_processor = tok
    if template_processor is None or not messages:
        out = get_chat_template(
            processor_any,
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **kwargs,
        )
        if isinstance(out, list):
            out = out[0].get("content", "") if out else ""
        return _vlm_normalize_thinking_newlines(str(out) if out else "")

    _vlm_diagnostics.used_prefix_stable = True

    def _apply(mes: List[Dict[str, Any]], gen: bool) -> str:
        try:
            raw = template_processor.apply_chat_template(
                mes,
                tokenize=False,
                add_generation_prompt=gen,
                **kwargs,
            )
        except Exception:
            return ""
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        return str(raw) if raw else ""

    parts = []
    try:
        prev = _apply([], False)
    except Exception:
        prev = ""
    for i in range(len(messages)):
        s = _apply(messages[: i + 1], False)
        if len(s) >= len(prev):
            parts.append(s[len(prev) :])
        prev = s
    full_no_gen = prev
    full_with_gen = _apply(messages, True)
    if full_with_gen.startswith(full_no_gen) and len(full_with_gen) >= len(full_no_gen):
        gen_suffix = full_with_gen[len(full_no_gen) :]
    else:
        gen_suffix = "\nAssistant:" if add_generation_prompt else ""
    return _vlm_normalize_thinking_newlines("".join(parts) + gen_suffix)


def _vlm_prompt_and_inputs(
    processor_any,
    config: Dict[str, Any],
    messages: List[Dict[str, Any]],
    images: List[Any],
    tools: Optional[Any] = None,
    enable_thinking: Optional[bool] = None,
    **kwargs: Any,
) -> Tuple[Any, Any, Any]:
    """
    Build formatted prompt and run prepare_inputs for VLM.
    Returns (input_ids, pixel_values, mask) where input_ids is mx.array; mask may be None.
    Uses prefix-stable formatting so cache prefix reuse works across turns.
    Pass tools= and enable_thinking= so the template output matches and cache aligns.
    """
    template_kwargs = dict(kwargs)
    if tools is not None:
        template_kwargs["tools"] = tools
    if enable_thinking is not None:
        template_kwargs["enable_thinking"] = enable_thinking
    formatted = _vlm_format_prompt_prefix_stable(
        processor_any, messages, add_generation_prompt=True, **template_kwargs
    )
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
    if input_ids is not None and hasattr(input_ids, "tolist"):
        input_ids = input_ids  # keep mx.array for now; convert to list where needed
    return input_ids, pixel_values, mask


def _vlm_sync_before_generation(pixel_values: Any, mask: Any) -> None:
    """
    Flush Metal work before VLM generation to prevent PyTorch and MLX from colliding.
    """
    try:
        import torch
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache() # CRITICAL: Force PyTorch to release all Metal encoders
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
    # Default to disabled to avoid long/self-referential planning loops unless explicitly requested.
    return False


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
) -> None:
    """
    Insert the standard full-turn cache entry and, for tool-call turns, also
    insert a prompt-only checkpoint. The checkpoint helps next-turn prefix reuse
    when provider-side tool-call serialization differs between turns.
    """
    # Always insert a prompt-only checkpoint when possible so the next turn can
    # match this prefix (cache=shorter). Required for VLM prefix reuse when the
    # session only has the full key (prompt+generated); also helps LM.
    if (
        generated_tokens
        and can_trim_prompt_cache(prompt_cache)
        and len(cache_key) > len(generated_tokens)
    ):
        try:
            prompt_only_cache = copy.deepcopy(prompt_cache)
            trim_prompt_cache(prompt_only_cache, len(generated_tokens))
            prompt_only_key = cache_key[: -len(generated_tokens)]
            PROMPT_CACHE.insert_cache(model_name, prompt_only_key, prompt_only_cache)
            SESSION_INDEX.register_cache_key(session_ctx, prompt_only_key)
        except Exception:
            pass

    PROMPT_CACHE.insert_cache(model_name, cache_key, prompt_cache)
    SESSION_INDEX.register_cache_key(session_ctx, cache_key)


def _normalize_prompt_for_cache(prompt):
    """
    Normalize volatile, non-semantic metadata that changes every request
    (e.g., inbound message_id) to keep cache keys stable across turns.
    """
    if not isinstance(prompt, str):
        return prompt
    if not SETTINGS.cache_canonicalize_tool_context:
        return prompt
    normalized = INBOUND_META_MESSAGE_ID_PATTERN.sub(
        r'\1__CACHE_STABLE_MESSAGE_ID__\2',
        prompt,
    )
    if INBOUND_TRUSTED_CONTEXT_BLOCK_PATTERN.search(normalized):
        normalized = INBOUND_TRUSTED_CONTEXT_BLOCK_PATTERN.sub(
            CACHE_STABLE_INBOUND_CONTEXT_BLOCK,
            normalized,
            count=1,
        )
    elif "# Project Context" in normalized:
        normalized = normalized.replace(
            "# Project Context",
            f"{CACHE_STABLE_INBOUND_CONTEXT_BLOCK}\n# Project Context",
            1,
        )
    normalized = INBOUND_CONTEXT_TO_PROJECT_BOUNDARY_PATTERN.sub(
        r"\1\n# Project Context",
        normalized,
        count=1,
    )
    return normalized


def _extract_session_context(body: Dict[str, Any], prompt_tokens: List[int]) -> SessionContext:
    def _read_any_id(container: Optional[Dict[str, Any]], keys: List[str]) -> Optional[str]:
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
        "repetition_penalty": body.get("repetition_penalty", SETTINGS.default_repetition_penalty),
        "repetition_context_size": body.get("repetition_context_size", SETTINGS.default_repetition_context_size),
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

def start_litellm_proxy():
    global proxy_process, proxy_config_path
    _terminal_status("🌉", f"Launching LiteLLM Proxy on port {SETTINGS.proxy_port}...")

    # Use proxy config so unsupported OpenAI params (e.g. "store") are dropped.
    config_yaml = f"""model_list:
  - model_name: {SETTINGS.proxy_model_id}
    litellm_params:
      model: {SETTINGS.proxy_model_id}
      api_base: http://127.0.0.1:{SETTINGS.mlx_port}/v1
      api_key: local
      # Keep these request fields when drop_params=true so this server can map
      # OpenClaw/Claude reasoning intent into tokenizer enable_thinking.
      allowed_openai_params:
        - reasoning_effort
litellm_settings:
  drop_params: true
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
        cmd,
        env=my_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE
    )
    time.sleep(SETTINGS.proxy_startup_wait_seconds)
    if proxy_process.poll() is not None:
        stderr_output = ""
        if proxy_process.stderr is not None:
            try:
                stderr_output = proxy_process.stderr.read().decode("utf-8", errors="replace")
            except Exception:
                stderr_output = ""
        raise RuntimeError(
            "LiteLLM proxy failed to start. "
            + (f"stderr: {stderr_output.strip()}" if stderr_output else "No stderr captured.")
        )
    _terminal_status("✅", f"LiteLLM Proxy ready at http://127.0.0.1:{SETTINGS.proxy_port}")

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

_terminal_status("🚀", f"Loading model: {SETTINGS.model_path}")
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
        from transformers.models.auto.configuration_auto import model_type_to_module_name

        def _patched_video_processor_class_from_name(class_name: str):
            for module_name, extractors in _vpa.VIDEO_PROCESSOR_MAPPING_NAMES.items():
                if extractors is not None and class_name in extractors:
                    mod_name = model_type_to_module_name(module_name)
                    module = importlib.import_module(f".{mod_name}", "transformers.models")
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
    model, processor = load_vlm(SETTINGS.model_path, tokenizer_config={"trust_remote_code": True})
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    is_vlm = True
    vlm_config = _config if isinstance(_config, dict) else getattr(model, "config", None)
    if vlm_config is None and hasattr(model, "config"):
        vlm_config = getattr(model.config, "__dict__", None) or {}
    _terminal_status("✅", "VLM model loaded (mlx-vlm).")
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
        _terminal_status("⚠️", "Torch acceleration: unknown (torch/torchvision not detected).")
else:
    model, tokenizer = load(SETTINGS.model_path, tokenizer_config={"trust_remote_code": True})
    _terminal_status("✅", "Model loaded (mlx-lm).")
    _terminal_status("⚡", "Torch acceleration: N/A (text-only model).")
_terminal_status(
    "🧠",
    (
        "Cache config loaded | "
        f"global_entries={SETTINGS.prompt_cache_max_entries_global} | "
        f"per_session_entries={SETTINGS.prompt_cache_max_entries_per_session} | "
        f"ttl_seconds={SETTINGS.prompt_cache_ttl_seconds} | "
        f"session_idle_seconds={SETTINGS.prompt_cache_session_max_idle_seconds} | "
        "session_partitioning=False (global only) | "
        f"canonicalize_tool_context={SETTINGS.cache_canonicalize_tool_context}"
    ),
)


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


def _stream_generate_unified(rest_tokens, max_tokens, sampler, prompt_cache, vlm_pixel_values=None, vlm_mask=None, cache_match_type="miss"):
    """
    Yields response objects with .text and .token (LM: GenerationResponse, VLM: GenerationResult).
    Dynamically checks for image tokens in rest_tokens to prevent Metal Segmentation Faults.
    """
    if is_vlm:
        # Prevent Segfault: We MUST pass vision tensors if rest_tokens contains image placeholders.
        # If we have a massive chunk of rest_tokens (e.g., > 100) on a 'shorter' hit, 
        # it almost certainly contains the system/user image tokens.
        has_image_tokens = False
        if rest_tokens:
            # Check for common Qwen3/GLM vision token IDs, or fallback to length heuristic
            vision_tokens = {151652, 151653, 151654, 151655} # Common Qwen vision IDs
            has_image_tokens = any(tok in vision_tokens for tok in rest_tokens)
            if not has_image_tokens and len(rest_tokens) > 100:
                has_image_tokens = True

        use_vision = (cache_match_type == "miss" or has_image_tokens) and vlm_pixel_values is not None
        
        rest_ids = mx.array([rest_tokens], dtype=mx.int32) if rest_tokens else mx.array([[0]], dtype=mx.int32)
        kwargs = {
            "input_ids": rest_ids,
            "pixel_values": vlm_pixel_values if use_vision else None,
            "mask": vlm_mask if use_vision else None,
            "prompt_cache": prompt_cache,
            "max_tokens": max_tokens,
            "sampler": sampler,
            "max_kv_size": SETTINGS.max_kv_size,
            "kv_group_size": SETTINGS.kv_group_size,
        }
        if SETTINGS.kv_bits is not None:
            kwargs["kv_bits"] = SETTINGS.kv_bits
        for resp in stream_generate_vlm(model, processor, "", image=None, **kwargs):
            yield resp
    else:
        for resp in stream_generate(**_stream_generate_kwargs(rest_tokens, max_tokens, sampler, prompt_cache)):
            yield resp

class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep terminal output focused on custom request lifecycle lines.
        return

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": SETTINGS.model_path,
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
        if self.path == "/v1/models":
            return self.do_GET()

        if self.path != "/v1/chat/completions":
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers["Content-Length"])
            body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except Exception:
            self.send_error(400, "Bad Request")
            return

        request_id = uuid.uuid4().hex[:12]
        tools = body.get("tools")
        reasoning_control = _extract_enable_thinking(body)
        enable_thinking = reasoning_control["enable_thinking"]

        vlm_pixel_values = None
        vlm_mask = None
        vlm_input_ids_raw = None

        if is_vlm:
            raw_messages = body.get("messages", [])
            session_ctx_early = _extract_session_context(body, [])
            stable_key = _vlm_stable_session_key(raw_messages) if raw_messages else None
            key = (SETTINGS.model_path, stable_key) if stable_key else None
            if (
                enable_thinking
                and len(raw_messages) > 0
                and raw_messages[-1].get("role") == "assistant"
            ):
                # Use stable key from first two messages so lookup matches store across
                # turns; implicit session_id is derived from prompt_tokens which we
                # don't have yet for VLM.
                with SESSION_FULL_ASSISTANT_LOCK:
                    stored = SESSION_FULL_ASSISTANT.get(key)
                # Use stored full assistant content when we have it so the tokenized prompt
                # matches the previous turn's cache (client may send stripped/different wording).
                use_stored = bool(stored)
                if use_stored:
                    messages_for_vlm = [dict(m) for m in raw_messages]
                    messages_for_vlm[-1]["content"] = stored
                    messages = _prepare_messages_for_vlm(messages_for_vlm, tools=tools)
                else:
                    messages = _prepare_messages_for_vlm(raw_messages, tools=tools)
            else:
                messages = _prepare_messages_for_vlm(raw_messages, tools=tools)
            images = _extract_images_from_messages(body.get("messages", []))
            with model_lock:
                vlm_input_ids_raw, vlm_pixel_values, vlm_mask = _vlm_prompt_and_inputs(
                    processor,
                    vlm_config or {},
                    messages,
                    images,
                    tools=tools,
                    enable_thinking=enable_thinking,
                )
                _vlm_sync_before_generation(vlm_pixel_values, vlm_mask)
            if vlm_input_ids_raw is not None:
                if hasattr(vlm_input_ids_raw, "flatten"):
                    prompt_tokens = vlm_input_ids_raw.flatten().tolist()
                else:
                    prompt_tokens = list(vlm_input_ids_raw)
            else:
                prompt_tokens = []
            # Token trajectory stitching: when we have stored token IDs for every assistant
            # turn, build prompt_tokens by stitching so the sequence matches the KV cache.
            assistant_count = sum(
                1 for m in messages if (m.get("role") or "").strip().lower() == "assistant"
            )
            if (
                enable_thinking
                and assistant_count > 0
                and stable_key is not None
            ):
                with TOKEN_TRAJECTORY_STORE_LOCK:
                    trajectory_list = list(TOKEN_TRAJECTORY_STORE.get(key, []))
                if trajectory_list:
                    use_count = min(len(trajectory_list), assistant_count)
                    trajectory_turns = trajectory_list[-use_count:]
                    stitched = _vlm_build_stitched_prompt_tokens(
                        processor,
                        messages,
                        trajectory_turns,
                        tools=tools,
                        enable_thinking=enable_thinking,
                        model_family=SETTINGS.model_family,
                    )
                    if stitched:
                        prompt_tokens = stitched
            # VLM cache key = this token list. We use _vlm_format_prompt_prefix_stable so
            # the prefix for the first K messages tokenizes identically across turns (cache=shorter).
            prompt = ""
        else:
            messages = _prepare_messages_for_template(body.get("messages", []))
            # Token trajectory stitching (LM path): stitch stored assistant token IDs when available
            prompt_tokens = None
            raw_messages_lm = body.get("messages", [])
            stable_key_lm = _vlm_stable_session_key(raw_messages_lm) if raw_messages_lm else None
            key_lm = (SETTINGS.model_path, stable_key_lm) if stable_key_lm else None
            assistant_count_lm = sum(
                1 for m in messages if (m.get("role") or "").strip().lower() == "assistant"
            )
            if (
                enable_thinking
                and assistant_count_lm > 0
                and key_lm is not None
            ):
                with TOKEN_TRAJECTORY_STORE_LOCK:
                    trajectory_list_lm = list(TOKEN_TRAJECTORY_STORE.get(key_lm, []))
                if trajectory_list_lm:
                    use_count_lm = min(len(trajectory_list_lm), assistant_count_lm)
                    trajectory_turns_lm = trajectory_list_lm[-use_count_lm:]
                    stitched_lm = _vlm_build_stitched_prompt_tokens(
                        tokenizer,
                        messages,
                        trajectory_turns_lm,
                        tools=tools,
                        enable_thinking=enable_thinking,
                        model_family=SETTINGS.model_family,
                    )
                    if stitched_lm:
                        prompt_tokens = stitched_lm
            if prompt_tokens is None:
                if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                    prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        tools=tools,
                        enable_thinking=enable_thinking,
                    )
                else:
                    prompt = messages[-1]["content"] if messages else ""
                cache_prompt = _normalize_prompt_for_cache(prompt)
                prompt_tokens = _tokenize_prompt(cache_prompt)
                prompt_was_normalized = cache_prompt != prompt
            else:
                prompt = ""
                prompt_was_normalized = False

        session_ctx = _extract_session_context(body, prompt_tokens)
        if is_vlm:
            prompt_was_normalized = bool(SETTINGS.cache_canonicalize_tool_context)
        cache_key = prompt_tokens[:]
        prompt_cache = None
        rest_tokens = prompt_tokens
        cache_session_tokens = prompt_tokens
        cache_match_type = "miss"
        matched_prefix_len = 0
        cache_selection_source = "none"
        rest_count = len(prompt_tokens)
        request_logger = None
        sampler, sampler_kwargs = _build_sampler(body)
        max_tokens = body.get("max_tokens", SETTINGS.default_max_tokens)
        is_streaming = body.get("stream", False)

        # --- INJECT DEBUG BLOCK HERE ---
        if SETTINGS.vlm_cache_debug and session_ctx.session_id:
            state = SESSION_INDEX._sessions.get(session_ctx.session_id)
            if state and state.keys:
                # Grab the most recent cache key for this session
                most_recent_stored_key = state.keys[-1]
                _debug_token_divergence(tokenizer, prompt_tokens, most_recent_stored_key, context_window=8)
        # --- END INJECT DEBUG BLOCK ---

        acquired = False
        generated_tokens = []
        generation_started_at = None
        first_token_at = None
        queue_started_at = time.time()
        try:
            model_lock.acquire(blocking=True)
            acquired = True
            
            generation_started_at = time.time()
            wait_seconds = generation_started_at - queue_started_at
            with prompt_cache_lock:
                (
                    prompt_cache,
                    rest_tokens,
                    cache_session_tokens,
                    cache_match_type,
                    matched_prefix_len,
                    cache_selection_source,
                ) = SESSION_INDEX.select_best_cache(
                    model_name=SETTINGS.model_path,
                    prompt_tokens=prompt_tokens,
                    session_ctx=session_ctx,
                    prompt_cache_store=PROMPT_CACHE,
                )
            if prompt_cache is None:
                cache_model = model.language_model if is_vlm and hasattr(model, "language_model") else model
                prompt_cache = make_prompt_cache(cache_model, max_kv_size=SETTINGS.max_kv_size)
                rest_tokens = prompt_tokens
            rest_count = len(rest_tokens) if rest_tokens is not None else len(prompt_tokens)
            cache_session_id = _cache_log_session_id(session_ctx, cache_session_tokens)
            if SETTINGS.enable_request_logging:
                try:
                    request_logger = CacheSessionTranscriptLogger(cache_session_id=cache_session_id)
                except Exception:
                    request_logger = None

            if request_logger:
                request_logger.log(
                    "prompt",
                    {
                        "request_meta": {
                            "path": self.path,
                            "stream": bool(body.get("stream", False)),
                            "model": body.get("model", SETTINGS.model_path),
                            "model_family": SETTINGS.model_family,
                            "temperature": body.get("temperature", SETTINGS.default_temperature),
                            "max_tokens": body.get("max_tokens", SETTINGS.default_max_tokens),
                            "enable_thinking": enable_thinking,
                            "thinking_source": reasoning_control["source"],
                            "thinking_raw": reasoning_control["raw"],
                            "cache_session_id": cache_session_id,
                            "cache_match_type": cache_match_type,
                            "cache_selection_source": cache_selection_source,
                            "matched_prefix_len": matched_prefix_len,
                            "cache_prompt_normalized": prompt_was_normalized,
                            "prompt_tokens": len(prompt_tokens),
                            "session_id": session_ctx.session_id,
                            "parent_session_id": session_ctx.parent_session_id,
                            "branch_id": session_ctx.branch_id,
                            "session_source": session_ctx.source,
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

            _terminal_status(
                "📨",
                f"Request {request_id} received | stream={body.get('stream', False)} | "
                f"prompt_tokens={len(prompt_tokens)} | cache={cache_match_type} | "
                f"prefix_tokens={matched_prefix_len} | rest_tokens={rest_count} | "
                f"normalized={prompt_was_normalized} | "
                f"thinking={enable_thinking} ({reasoning_control['source']}) | "
                f"family={SETTINGS.model_family} | session={session_ctx.session_id} ({cache_selection_source})",
            )
            _terminal_status(
                "⚙️",
                f"Request {request_id} generation started | wait={wait_seconds:.2f}s | "
                f"cache={cache_match_type} | prefix_tokens={matched_prefix_len} | prefill_tokens={rest_count}",
                indent=1,
            )

            if not is_streaming:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()

                generated_parts = []
                progress_last_at = time.time()
                for response in _stream_generate_unified(
                    rest_tokens, max_tokens, sampler, prompt_cache,
                    vlm_pixel_values=vlm_pixel_values, vlm_mask=vlm_mask, cache_match_type=cache_match_type,
                ):
                    generated_parts.append(response.text)
                    generated_tokens.append(int(response.token))
                    if first_token_at is None:
                        first_token_at = time.time()
                    if len(generated_tokens) % 64 == 0 and (time.time() - progress_last_at) >= 1.0:
                        progress_last_at = time.time()
                        _terminal_status(
                            "⏳",
                            f"Request {request_id} in progress | generated_tokens={len(generated_tokens)}",
                            indent=1,
                        )
                response_text = "".join(generated_parts)
                raw_response_text = response_text
                response_text = _normalize_assistant_text(
                    response_text, enable_thinking, SETTINGS.model_family
                )
                message_text, tool_calls = _extract_openai_tool_calls(
                    response_text, SETTINGS.model_family
                )
                # Hide <think> blocks from the client whenever reasoning was requested (LM or VLM).
                if enable_thinking:
                    message_text = _strip_thinking_from_content(message_text)
                if enable_thinking:
                    traj_key = (SETTINGS.model_path, _vlm_stable_session_key(body.get("messages", [])))
                    _token_trajectory_append(traj_key, generated_tokens)
                if is_vlm and enable_thinking:
                    with SESSION_FULL_ASSISTANT_LOCK:
                        traj_key = (SETTINGS.model_path, _vlm_stable_session_key(body.get("messages", [])))
                        SESSION_FULL_ASSISTANT[traj_key] = raw_response_text
                        SESSION_FULL_ASSISTANT.move_to_end(traj_key, last=True)
                        while len(SESSION_FULL_ASSISTANT) > MAX_SESSION_FULL_ASSISTANT:
                            SESSION_FULL_ASSISTANT.popitem(last=False)
                finish_reason = "tool_calls" if tool_calls else "stop"
                cache_key.extend(generated_tokens)
                with prompt_cache_lock:
                    _insert_cache_entries(
                        model_name=SETTINGS.model_path,
                        session_ctx=session_ctx,
                        cache_key=cache_key,
                        prompt_cache=prompt_cache,
                        generated_tokens=generated_tokens,
                        tool_calls=tool_calls,
                    )

                response_id = f"chatcmpl-{int(time.time())}"
                full_response = {
                    "id": response_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": SETTINGS.model_path,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": message_text
                        },
                        "finish_reason": finish_reason
                    }],
                    "usage": { 
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                }
                if tool_calls:
                    full_response["choices"][0]["message"]["tool_calls"] = tool_calls
                self.wfile.write(json.dumps(full_response).encode("utf-8"))
                if request_logger:
                    timing = {}
                    if first_token_at is not None and generation_started_at is not None:
                        timing["prefill_seconds"] = first_token_at - generation_started_at
                        timing["decode_seconds"] = time.time() - first_token_at
                        timing["prefill_tps"] = rest_count / timing["prefill_seconds"] if timing["prefill_seconds"] > 0 else None
                        timing["decode_tps"] = len(generated_tokens) / timing["decode_seconds"] if timing["decode_seconds"] > 0 else None
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
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                response_id = f"chatcmpl-{int(time.time())}"
                role_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": SETTINGS.model_path,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                }
                self.wfile.write(f"data: {json.dumps(role_chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()

                raw_parts = []
                progress_last_at = time.time()
                for response in _stream_generate_unified(
                    rest_tokens, max_tokens, sampler, prompt_cache,
                    vlm_pixel_values=vlm_pixel_values, vlm_mask=vlm_mask, cache_match_type=cache_match_type,
                ):
                    generated_tokens.append(int(response.token))
                    if first_token_at is None:
                        first_token_at = time.time()
                    response_text = response.text
                    if response_text:
                        raw_parts.append(response_text)
                    if len(generated_tokens) % 64 == 0 and (time.time() - progress_last_at) >= 1.0:
                        progress_last_at = time.time()
                        _terminal_status(
                            "⏳",
                            f"Request {request_id} in progress | generated_tokens={len(generated_tokens)}",
                            indent=1,
                        )

                full_text = "".join(raw_parts)
                raw_full_text = full_text
                full_text = _normalize_assistant_text(
                    full_text, enable_thinking, SETTINGS.model_family
                )
                message_text, tool_calls = _extract_openai_tool_calls(
                    full_text, SETTINGS.model_family
                )
                # Hide <think> blocks from the client whenever reasoning was requested (LM or VLM).
                if enable_thinking:
                    message_text = _strip_thinking_from_content(message_text)
                if enable_thinking:
                    traj_key_stream = (SETTINGS.model_path, _vlm_stable_session_key(body.get("messages", [])))
                    _token_trajectory_append(traj_key_stream, generated_tokens)
                if is_vlm and enable_thinking:
                    with SESSION_FULL_ASSISTANT_LOCK:
                        traj_key_stream = (SETTINGS.model_path, _vlm_stable_session_key(body.get("messages", [])))
                        SESSION_FULL_ASSISTANT[traj_key_stream] = raw_full_text
                        SESSION_FULL_ASSISTANT.move_to_end(traj_key_stream, last=True)
                        while len(SESSION_FULL_ASSISTANT) > MAX_SESSION_FULL_ASSISTANT:
                            SESSION_FULL_ASSISTANT.popitem(last=False)
                cache_key.extend(generated_tokens)
                with prompt_cache_lock:
                    _insert_cache_entries(
                        model_name=SETTINGS.model_path,
                        session_ctx=session_ctx,
                        cache_key=cache_key,
                        prompt_cache=prompt_cache,
                        generated_tokens=generated_tokens,
                        tool_calls=tool_calls,
                    )

                if message_text:
                    chunk = {
                        "id": response_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": SETTINGS.model_path,
                        "choices": [{"index": 0, "delta": {"content": message_text}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                if tool_calls:
                    for idx, tc in enumerate(tool_calls):
                        tc_chunk = {
                            "id": response_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": SETTINGS.model_path,
                            "choices": [{
                                "index": 0,
                                "delta": {"tool_calls": [{**tc, "index": idx}]},
                                "finish_reason": None,
                            }],
                        }
                        self.wfile.write(f"data: {json.dumps(tc_chunk)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                finish_reason = "tool_calls" if tool_calls else "stop"
                final_chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": SETTINGS.model_path,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
                }
                self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                if request_logger:
                    timing = {}
                    if first_token_at is not None and generation_started_at is not None:
                        timing["prefill_seconds"] = first_token_at - generation_started_at
                        timing["decode_seconds"] = time.time() - first_token_at
                        timing["prefill_tps"] = rest_count / timing["prefill_seconds"] if timing["prefill_seconds"] > 0 else None
                        timing["decode_tps"] = len(generated_tokens) / timing["decode_seconds"] if timing["decode_seconds"] > 0 else None
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
            _terminal_status("⚠️", f"Request {request_id} client disconnected (BrokenPipeError)", indent=1)
            if request_logger:
                request_logger.log("generation", "client disconnected (BrokenPipeError)", request_id=request_id)
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
                if first_token_at is not None:
                    prefill_seconds = first_token_at - generation_started_at
                    decode_seconds = max(end_at - first_token_at, 1e-9)
                    decode_tps = output_tokens / decode_seconds if output_tokens else 0.0
                    prefill_tps = rest_count / prefill_seconds if prefill_seconds > 0 else 0.0
                    _terminal_status(
                        "✅",
                        (
                            f"Request {request_id} finished | output_tokens={output_tokens} | "
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
                            f"Request {request_id} finished | output_tokens={output_tokens} | "
                            f"elapsed={elapsed:.2f}s | tok/s={speed:.2f}"
                        ),
                        indent=1,
                    )
            # On normal exit or Python exception, release the lock. On process abort (e.g. Metal
            # "uncommitted encoder" crash), finally may not run, so the "leaked semaphore" warning
            # at shutdown is expected; fixing the Metal crash resolves it.
            if acquired:
                model_lock.release()

def run():
    start_litellm_proxy()
    server_address = (SETTINGS.mlx_host, SETTINGS.mlx_port)
    httpd = ThreadingHTTPServer(server_address, APIHandler)

    print("\n" + "=" * 50)
    print("🟢 SYSTEM READY")
    print(f"   • Mode:         {'VLM (vision)' if is_vlm else 'LM (text-only)'}")
    print(f"   • MLX Engine:   http://{SETTINGS.mlx_host}:{SETTINGS.mlx_port}")
    print(f"   • OpenClaw Link: http://127.0.0.1:{SETTINGS.proxy_port}")
    print("=" * 50 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    run()