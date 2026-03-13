#!/usr/bin/env python3
import sys
import json
import time
import requests
import hashlib
import random
import string
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
SERVER_URL = "http://127.0.0.1:4000/v1/chat/completions"
LOG_ROOT = Path("logs")


def get_session_log_path(session_id):
    today = datetime.now().strftime("%Y-%m-%d")
    hashed_id = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:16]
    return LOG_ROOT / today / f"cache-session-{hashed_id}.log"


def tail_log_entries(log_path, last_pos=0):
    entries = []
    if not log_path.exists():
        return entries, last_pos

    with log_path.open("r", encoding="utf-8") as f:
        f.seek(last_pos)
        content = f.read()
        last_pos = f.tell()

    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "direction=prompt" in line:
            # found a header
            # consume lines until empty line or next header
            json_lines = []
            i += 1
            while i < len(lines):
                if lines[i].strip() == "" or lines[i].startswith("["):
                    # empty line or next header
                    break
                json_lines.append(lines[i])
                i += 1

            if json_lines:
                try:
                    json_str = "\n".join(json_lines)
                    data = json.loads(json_str)
                    meta = data.get("request_meta", {})
                    entries.append(meta)
                except Exception as e:
                    print(f"Error parsing log entry: {e}")
            continue
        i += 1

    return entries, last_pos


def run_probe(scenario_name, mutation_func):
    session_id = f"probe-{scenario_name}-{int(time.time())}"
    log_path = get_session_log_path(session_id)
    print(f"\n--- Scenario: {scenario_name} (Session: {session_id}) ---")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. " * 50,
        },  # ~300 tokens
        {"role": "user", "content": "Count from 1 to 5. " * 20},  # ~120 tokens
    ]

    # Turn 1
    print("Turn 1: Initial request")
    try:
        resp = requests.post(
            SERVER_URL,
            json={
                "model": "openai/local",
                "messages": messages,
                "session_id": session_id,
                "stream": False,
                "max_tokens": 10,
            },
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"Request failed: {e}")
        return

    time.sleep(2)  # Wait for log flush
    entries, last_pos = tail_log_entries(log_path)
    if not entries:
        print(f"Warning: No log entries found at {log_path}")
    else:
        last = entries[-1]
        print(
            f"Turn 1 Cache: {last.get('cache_match_type')} (prefix: {last.get('matched_prefix_len')}/{last.get('prompt_tokens')})"
        )

    # Turn 2: Apply mutation
    print("Turn 2: Mutated request")
    # Simulate assistant response from Turn 1
    messages.append({"role": "assistant", "content": "1, 2, 3, 4, 5"})

    # Apply mutation
    messages = mutation_func(messages)

    # New user message
    messages.append({"role": "user", "content": "Now count to 10."})

    try:
        resp = requests.post(
            SERVER_URL,
            json={
                "model": "openai/local",
                "messages": messages,
                "session_id": session_id,
                "stream": False,
                "max_tokens": 10,
            },
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"Request failed: {e}")
        return

    time.sleep(2)
    entries, _ = tail_log_entries(log_path, last_pos)
    if not entries:
        print("Warning: No log entries found for Turn 2.")
    else:
        last = entries[-1]
        hit_ratio = 0.0
        prompt_tokens = last.get("prompt_tokens", 0)
        matched_prefix = last.get("matched_prefix_len", 0)

        if prompt_tokens > 0:
            hit_ratio = matched_prefix / prompt_tokens

        print(
            f"Turn 2 Cache: {last.get('cache_match_type')} (prefix: {matched_prefix}/{prompt_tokens}) - Hit Ratio: {hit_ratio:.2f}"
        )

        if hit_ratio < 0.5:  # Anomaly threshold
            print("ANOMALY DETECTED! hit_ratio < 0.5")
            print("Dumping diff...")
            subprocess.run([sys.executable, "scripts/diff_turns.py", str(log_path)])


# Scenarios


def scenario_normal_append(msgs):
    # No mutation, just standard append
    return msgs


def scenario_whitespace_drift(msgs):
    # FP-1: Simulate trailing whitespace added to system prompt (the real OpenCode trigger).
    # The _normalize_message_content_for_diff() should absorb this so stable_prefix_msg_count >= 2.
    msgs[0]["content"] = msgs[0]["content"] + " "
    return msgs


def scenario_timestamp_drift(msgs):
    # Semantic mutation (not whitespace). Used to confirm normalisation does NOT
    # absorb real content changes. stable_prefix_msg_count should be 0.
    msgs[0]["content"] = msgs[0]["content"].replace("assistant", "robot")
    return msgs


def scenario_mid_insertion(msgs):
    # Insert a message in the middle
    # Insert before the assistant response
    msgs.insert(1, {"role": "user", "content": "Wait, also check this."})
    return msgs


if __name__ == "__main__":
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        if scenario == "normal":
            run_probe("normal", scenario_normal_append)
        elif scenario == "drift":
            # FP-1: trailing whitespace — normalisation should absorb this
            run_probe("drift", scenario_whitespace_drift)
        elif scenario == "semantic":
            # Semantic mutation — normalisation must NOT absorb this
            run_probe("semantic", scenario_timestamp_drift)
        elif scenario == "insert":
            run_probe("insert", scenario_mid_insertion)
        else:
            print("Unknown scenario")
    else:
        run_probe("normal", scenario_normal_append)
        run_probe("drift", scenario_whitespace_drift)
        run_probe("semantic", scenario_timestamp_drift)
        run_probe("insert", scenario_mid_insertion)
