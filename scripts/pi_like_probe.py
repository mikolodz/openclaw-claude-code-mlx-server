#!/usr/bin/env python3
"""
pi-like multi-turn probe: mimics a pi-style agent loop where the model produces
tool_calls, the harness mocks tool execution, and the conversation grows by
appending assistant + tool messages before each new turn.

Purpose: reproduce VLM-path cache regression that shows up only with real tool
traffic (cannot be triggered by probe_session.py, which has no tools).

Reports per-turn cache metrics by tailing the server's cache-session log.

Usage:
  python scripts/pi_like_probe.py [--with-image PATH] [--turns N] [--url URL]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

SERVER_URL = "http://127.0.0.1:4000/v1/chat/completions"
MODEL_ID = "openai/local"
LOG_ROOT = Path("logs")

SYSTEM_PROMPT = (
    "You are a helpful coding assistant. Use the provided tools to help the user. "
    "Always call one tool at a time and wait for the result before proceeding. "
    "When all steps are complete, respond with a short summary in plain text."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a bash command.",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "description": "Bash command"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file.",
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    },
]

# Scripted tool-result responses keyed by tool+key for reproducibility.
MOCK_RESULTS = {
    ("read_file", "src/hello.txt"):
        "Hello world!\nThis is a sample file.\nLine 3 here.\nLine 4 here.\n",
    ("run_bash", "wc -l src/hello.txt"): "       4 src/hello.txt",
    ("run_bash", "ls src"): "hello.txt\nsummary.txt",
    ("write_file", "src/summary.txt"): "File written OK.",
}


def mock_tool(name: str, args: dict) -> str:
    if name == "read_file":
        return MOCK_RESULTS.get((name, args.get("path", "")), "FILE_NOT_FOUND")
    if name == "run_bash":
        return MOCK_RESULTS.get((name, args.get("command", "").strip()), "OK")
    if name == "write_file":
        return MOCK_RESULTS.get((name, args.get("path", "")), "WROTE")
    return "NO_SUCH_TOOL"


def load_image_as_data_url(path: str) -> str:
    data = Path(path).read_bytes()
    ext = Path(path).suffix.lstrip(".").lower() or "png"
    b64 = base64.b64encode(data).decode()
    return f"data:image/{ext};base64,{b64}"


def get_session_log_path(session_id: str) -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    h = hashlib.sha1(session_id.encode()).hexdigest()[:16]
    return LOG_ROOT / day / f"cache-session-{h}.log"


def tail_last_meta(log_path: Path, pos: int) -> tuple[dict | None, int]:
    if not log_path.exists():
        return None, pos
    with log_path.open("r", encoding="utf-8") as f:
        f.seek(pos)
        content = f.read()
        new_pos = f.tell()
    lines = content.splitlines()
    metas = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "direction=prompt" in line:
            j = i + 1
            buf = []
            while j < len(lines) and lines[j].strip() != "" and not lines[j].startswith("["):
                buf.append(lines[j])
                j += 1
            if buf:
                try:
                    data = json.loads("\n".join(buf))
                    metas.append(data.get("request_meta", {}))
                except Exception:
                    pass
            i = j
        else:
            i += 1
    return (metas[-1] if metas else None), new_pos


def _parse_stream(resp):
    """Collect streaming SSE chunks into a single OpenAI-style response."""
    content_parts = []
    tool_calls = {}
    finish = "stop"
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        if raw.startswith(":"):
            continue
        if raw.startswith("data: "):
            raw = raw[6:]
        if raw.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(raw)
        except Exception:
            continue
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_calls.setdefault(idx, {"id": tc.get("id"), "type": tc.get("type", "function"), "function": {"name": "", "arguments": ""}})
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] = fn["name"]
            if fn.get("arguments") is not None:
                slot["function"]["arguments"] += fn["arguments"]
            if tc.get("id"):
                slot["id"] = tc["id"]
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    msg = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls:
        msg["tool_calls"] = [tool_calls[k] for k in sorted(tool_calls.keys())]
    return msg, finish


def run_turn(session_id: str, messages, tools, log_path: Path, pos: int, turn_label: str, stream: bool = False):
    resp = requests.post(
        SERVER_URL,
        json={
            "model": MODEL_ID,
            "messages": messages,
            "tools": tools,
            "session_id": session_id,
            "stream": stream,
            "max_tokens": 512,
            "temperature": 0.2,
        },
        timeout=300,
        stream=stream,
    )
    resp.raise_for_status()
    if stream:
        msg, finish = _parse_stream(resp)
    else:
        out = resp.json()
        msg = out["choices"][0]["message"]
        finish = out["choices"][0].get("finish_reason", "stop")
    time.sleep(0.5)
    meta, new_pos = tail_last_meta(log_path, pos)
    if meta:
        pt = meta.get("prompt_tokens", 0)
        mp = meta.get("matched_prefix_len", 0)
        ratio = (mp / pt * 100) if pt else 0.0
        print(
            f"  [{turn_label}] cache={meta.get('cache_match_type')} "
            f"{mp}/{pt} ({ratio:.1f}%) src={meta.get('cache_selection_source')} "
            f"stable={meta.get('stable_prefix_msg_count')} msgs / "
            f"{meta.get('stable_prefix_token_len')} toks"
        )
    else:
        print(f"  [{turn_label}] (no log meta yet)")
    return msg, finish, new_pos


def first_user_message(user_text: str, image_path: str | None):
    if image_path:
        return {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": load_image_as_data_url(image_path)}},
                {"type": "text", "text": user_text},
            ],
        }
    return {"role": "user", "content": user_text}


def agent_loop(session_id: str, user_text: str, image_path: str | None, max_turns: int, stream: bool = False):
    log_path = get_session_log_path(session_id)
    pos = log_path.stat().st_size if log_path.exists() else 0
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        first_user_message(user_text, image_path),
    ]
    for turn in range(1, max_turns + 1):
        msg, finish, pos = run_turn(session_id, messages, TOOLS, log_path, pos, f"T{turn}", stream=stream)
        # Append assistant
        asst = {"role": "assistant", "content": msg.get("content", "") or ""}
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            asst["tool_calls"] = tool_calls
        messages.append(asst)
        if finish != "tool_calls" or not tool_calls:
            print(f"  [T{turn}] finish={finish}; no tool calls — agent loop done")
            break
        # Execute each tool call, append tool result
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            result = mock_tool(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            })
    return messages


def main():
    global SERVER_URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--with-image", default=None, help="Optional image file attached to first turn")
    ap.add_argument("--url", default=SERVER_URL)
    ap.add_argument("--session", default=None)
    ap.add_argument("--stream", action="store_true", help="Request server-sent streaming like pi does")
    args = ap.parse_args()

    SERVER_URL = args.url

    session_id = args.session or f"probe-pi-like-{int(time.time())}"
    print(f"=== pi-like probe | session={session_id} | image={args.with_image or 'none'} | turns={args.turns} ===")
    if args.with_image:
        user_text = (
            "I've attached a diagram. You MUST complete all of these steps using tools — "
            "do not answer in plain text until every step has executed a tool:\n"
            "1. Call read_file on 'src/hello.txt'.\n"
            "2. Call run_bash with command 'wc -l src/hello.txt'.\n"
            "3. Call run_bash with command 'ls src'.\n"
            "4. Call write_file to create 'src/summary.txt' whose content is "
            "'File has N lines' (substitute N from step 2).\n"
            "5. After step 4, reply with a one-sentence summary that also briefly mentions "
            "the diagram contents."
        )
    else:
        user_text = (
            "You MUST complete all of these steps using tools — do not answer in plain text "
            "until every step has executed a tool:\n"
            "1. Call read_file on 'src/hello.txt'.\n"
            "2. Call run_bash with command 'wc -l src/hello.txt'.\n"
            "3. Call run_bash with command 'ls src'.\n"
            "4. Call write_file to create 'src/summary.txt' with content "
            "'File has N lines' (N from step 2).\n"
            "5. After step 4, reply with a one-sentence summary."
        )
    agent_loop(session_id, user_text, args.with_image, args.turns, stream=args.stream)


if __name__ == "__main__":
    main()
