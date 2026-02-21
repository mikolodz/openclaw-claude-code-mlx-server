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
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dotenv import load_dotenv
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.models.cache import make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache

SCRIPT_DIR = Path(__file__).resolve().parent
DOTENV_PATH = SCRIPT_DIR / ".env"
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
    normalize_write_tool_content_for_prompt: bool
    cache_canonicalize_tool_context: bool
    cache_session_partitioning: bool
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
        normalize_write_tool_content_for_prompt=_env_bool("NORMALIZE_WRITE_TOOL_CONTENT_FOR_PROMPT", True),
        cache_canonicalize_tool_context=_env_bool_any(
            ["CACHE_CANONICALIZE_TOOL_CONTEXT"],
            True,
        ),
        cache_session_partitioning=_env_bool_any(
            ["CACHE_SESSION_PARTITIONING"],
            True,
        ),
        log_root=Path(_env_str("LOG_ROOT", str(SCRIPT_DIR / "logs"))),
        proxy_startup_wait_seconds=_env_float("PROXY_STARTUP_WAIT_SECONDS", 2.0),
        proxy_model_id=proxy_model_id,
    )


SETTINGS = _build_settings()

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


def _terminal_status(icon: str, message: str, indent: int = 0) -> None:
    with console_lock:
        ts = datetime.now().strftime("%H:%M:%S")
        pad = "  " * max(indent, 0)
        print(f"{pad}{icon} [{ts}] {message}", flush=True)

class LRUPromptCache:
    @dataclass
    class CacheEntry:
        prompt_cache: List[Any]
        count: int
        touched_at: float

    @dataclass
    class SearchResult:
        model: Any
        exact: List[int]
        shorter: List[int]
        longer: List[int]
        common_prefix: int
        matched_prefix_len: int = 0

    def __init__(self, max_size=10, ttl_seconds=1800):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._lru = deque()

    def _is_expired(self, entry):
        # TTL <= 0 means "no TTL expiry" (keep entries until LRU pressure).
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - entry.touched_at) > self.ttl_seconds

    def _prune_expired(self):
        stale = []
        for model, tokens in list(self._lru):
            try:
                entry = self._get(model, tokens)
            except Exception:
                continue
            if self._is_expired(entry):
                stale.append((model, tokens))
        for model, tokens in stale:
            try:
                self._delete(model, tokens)
            except Exception:
                pass
            try:
                self._lru.remove((model, tokens))
            except ValueError:
                pass

    def prune_expired(self):
        self._prune_expired()

    def _search(self, model, tokens):
        tokens = tuple(tokens)
        if model not in self._cache:
            return self.SearchResult(model, None, None, None, 0, 0)

        current = self._cache[model]
        last_cache_index = -1
        index = 0

        while index < len(tokens) and tokens[index] in current:
            current = current[tokens[index]]
            if "cache" in current:
                last_cache_index = index
            index += 1

        if last_cache_index == len(tokens) - 1:
            return self.SearchResult(model, tokens, None, None, 0, len(tokens))

        shorter = None
        if last_cache_index > 0:
            shorter = tokens[: last_cache_index + 1]

        longer = None
        common_prefix = index
        if index > 0 and last_cache_index <= 0:
            best = None
            stack = [(current, tuple())]
            while stack:
                node, extra = stack.pop()
                if "cache" in node:
                    if best is None or len(extra) < len(best):
                        best = extra
                else:
                    for tok in node:
                        stack.append((node[tok], extra + (tok,)))
            if best is not None:
                longer = tokens[:index] + best

        matched_prefix_len = len(shorter) if shorter is not None else common_prefix
        return self.SearchResult(model, None, shorter, longer, common_prefix, matched_prefix_len)

    def _get(self, model, tokens):
        tokens = tuple(tokens)
        current = self._cache[model]
        for tok in tokens:
            current = current[tok]
        return current["cache"]

    def _delete(self, model, tokens):
        tokens = tuple(tokens)
        path = [self._cache[model]]
        for tok in tokens:
            path.append(path[-1][tok])
        del path[-1]["cache"]
        for i in reversed(range(len(tokens))):
            prev_node, node, tok = path[i], path[i + 1], tokens[i]
            if len(node) > 0:
                break
            del prev_node[tok]

    def _extract(self, model, tokens):
        tokens = tuple(tokens)
        entry = self._get(model, tokens)
        entry.touched_at = time.time()
        if entry.count == 1:
            self._delete(model, tokens)
            self._lru.remove((model, tokens))
            return entry
        entry.count -= 1
        return self.CacheEntry(copy.deepcopy(entry.prompt_cache), 1, time.time())

    def contains_tokens(self, model, tokens):
        tokens = tuple(tokens)
        if model not in self._cache:
            return False
        current = self._cache[model]
        for tok in tokens:
            if tok not in current:
                return False
            current = current[tok]
        return "cache" in current

    def extract_exact_cache(self, model, tokens):
        tokens = tuple(tokens)
        if not self.contains_tokens(model, tokens):
            return None
        return self._extract(model, tokens)

    def fetch_nearest_cache(self, model, tokens):
        """Returns (prompt_cache, rest_tokens, cache_session_tokens, cache_match_type, matched_prefix_len).
        cache_match_type is one of 'exact', 'shorter', 'longer', 'miss'.
        """
        self._prune_expired()
        result = self._search(model, tokens)

        if result.exact is not None:
            entry = self._extract(result.model, result.exact)
            # Keep one token to continue decoding safely.
            if len(tokens) > 1 and can_trim_prompt_cache(entry.prompt_cache):
                trim_prompt_cache(entry.prompt_cache, 1)
                return entry.prompt_cache, tokens[-1:], result.exact, "exact", len(tokens) - 1
            return entry.prompt_cache, tokens, result.exact, "exact", len(tokens)

        if result.shorter is not None:
            entry = self._extract(result.model, result.shorter)
            prefix_len = len(result.shorter)
            return entry.prompt_cache, tokens[prefix_len:], result.shorter, "shorter", prefix_len

        if result.longer is not None:
            entry = self._get(result.model, result.longer)
            if not can_trim_prompt_cache(entry.prompt_cache):
                return None, tokens, tokens, "miss", 0
            prefix = min(len(tokens) - 1, result.common_prefix)
            # Avoid deepcopy: extract (remove from cache), then trim in place.
            entry = self._extract(result.model, result.longer)
            num_to_trim = len(result.longer) - prefix
            trim_prompt_cache(entry.prompt_cache, num_to_trim)
            return entry.prompt_cache, tokens[prefix:], result.longer, "longer", prefix

        return None, tokens, tokens, "miss", 0

    def insert_cache(self, model, tokens, prompt_cache):
        tokens = tuple(tokens)
        self._prune_expired()
        if model not in self._cache:
            self._cache[model] = {}

        current = self._cache[model]
        for tok in tokens:
            if tok not in current:
                current[tok] = {}
            current = current[tok]

        if "cache" in current:
            current["cache"].count += 1
            current["cache"].touched_at = time.time()
            self._lru.remove((model, tokens))
        else:
            current["cache"] = self.CacheEntry(prompt_cache, 1, time.time())

        self._lru.append((model, tokens))
        if len(self._lru) > self.max_size:
            old_model, old_tokens = self._lru.popleft()
            self._delete(old_model, old_tokens)

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
        prompt_cache_store.prune_expired()
        self._prune_idle()
        if not SETTINGS.cache_session_partitioning:
            selected = prompt_cache_store.fetch_nearest_cache(model_name, prompt_tokens)
            return (*selected, "global_no_partition")

        best = None
        chain = self._lineage_chain(session_ctx.session_id)
        for depth, sid in enumerate(chain):
            state = self._sessions.get(sid)
            if state is None:
                continue
            state.touched_at = time.time()
            # Search fresh keys first, then long-lived anchors.
            candidates: List[Tuple[int, ...]] = []
            seen_candidates = set()
            for cache_tokens in list(reversed(state.keys)) + list(reversed(state.anchors)):
                if cache_tokens in seen_candidates:
                    continue
                seen_candidates.add(cache_tokens)
                candidates.append(cache_tokens)

            for recency_rank, cache_tokens in enumerate(candidates):
                if not prompt_cache_store.contains_tokens(model_name, cache_tokens):
                    continue
                prefix_len = self._lcp_len(prompt_tokens, cache_tokens)
                if prefix_len <= 0:
                    continue
                score = (prefix_len, -depth, -recency_rank)
                if best is None or score > best["score"]:
                    best = {
                        "score": score,
                        "cache_tokens": cache_tokens,
                        "depth": depth,
                        "session_id": sid,
                    }

        if best is not None:
            cache_entry = prompt_cache_store.extract_exact_cache(model_name, best["cache_tokens"])
            if cache_entry is not None:
                selection = self._selection_from_exact_entry(
                    prompt_tokens=prompt_tokens,
                    cache_tokens=best["cache_tokens"],
                    cache_entry=cache_entry,
                )
                if selection is not None:
                    source = (
                        "session_local"
                        if best["depth"] == 0
                        else f"session_ancestor_depth_{best['depth']}"
                    )
                    return (*selection, source)

        selected = prompt_cache_store.fetch_nearest_cache(model_name, prompt_tokens)
        return (*selected, "global_fallback")


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

_terminal_status("🚀", f"Loading model: {SETTINGS.model_path}")
model, tokenizer = load(SETTINGS.model_path, tokenizer_config={"trust_remote_code": True})
_terminal_status("✅", "Model loaded.")
_terminal_status(
    "🧠",
    (
        "Cache config loaded | "
        f"global_entries={SETTINGS.prompt_cache_max_entries_global} | "
        f"per_session_entries={SETTINGS.prompt_cache_max_entries_per_session} | "
        f"ttl_seconds={SETTINGS.prompt_cache_ttl_seconds} | "
        f"session_idle_seconds={SETTINGS.prompt_cache_session_max_idle_seconds} | "
        f"session_partitioning={SETTINGS.cache_session_partitioning} | "
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
        messages = _prepare_messages_for_template(body.get("messages", []))
        tools = body.get("tools")
        reasoning_control = _extract_enable_thinking(body)
        enable_thinking = reasoning_control["enable_thinking"]

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
        session_ctx = _extract_session_context(body, prompt_tokens)
        prompt_was_normalized = cache_prompt != prompt
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
                prompt_cache = make_prompt_cache(model, max_kv_size=SETTINGS.max_kv_size)
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
                        },
                        "messages": messages,
                        "tools": tools,
                        "rendered_prompt": prompt,
                    },
                    request_id=request_id,
                )
                request_logger.log(
                    "sampler",
                    {
                        "applied_kwargs": sampler_kwargs,
                        "rest_tokens": rest_count,
                        "matched_prefix_len": matched_prefix_len,
                    },
                    request_id=request_id,
                )

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
                for response in stream_generate(**_stream_generate_kwargs(rest_tokens, max_tokens, sampler, prompt_cache)):
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
                finish_reason = "tool_calls" if tool_calls else "stop"
                cache_key.extend(generated_tokens)
                with prompt_cache_lock:
                    PROMPT_CACHE.insert_cache(SETTINGS.model_path, cache_key, prompt_cache)
                    SESSION_INDEX.register_cache_key(session_ctx, cache_key)

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
                for response in stream_generate(**_stream_generate_kwargs(rest_tokens, max_tokens, sampler, prompt_cache)):
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
                cache_key.extend(generated_tokens)
                with prompt_cache_lock:
                    PROMPT_CACHE.insert_cache(SETTINGS.model_path, cache_key, prompt_cache)
                    SESSION_INDEX.register_cache_key(session_ctx, cache_key)

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
            if acquired:
                model_lock.release()

def run():
    start_litellm_proxy()
    server_address = (SETTINGS.mlx_host, SETTINGS.mlx_port)
    httpd = ThreadingHTTPServer(server_address, APIHandler)

    print("\n" + "=" * 50)
    print("🟢 SYSTEM READY")
    print(f"   • MLX Engine:   http://{SETTINGS.mlx_host}:{SETTINGS.mlx_port}")
    print(f"   • OpenClaw Link: http://127.0.0.1:{SETTINGS.proxy_port}")
    print("=" * 50 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    run()