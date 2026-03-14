#!/usr/bin/env python3
import sys
import json
import re
import difflib
from pathlib import Path
from typing import List, Dict, Any, Tuple

def parse_log_file(log_path: Path) -> List[Dict[str, Any]]:
    """
    Parses a server log file and returns a list of request objects.
    Each object contains the timestamp, request_id, and the parsed JSON payload
    for 'direction=prompt'.
    """
    entries = []
    current_entry = {}
    
    # Regex to capture the log line header
    # Example: [2026-03-12T10:00:00.000000] request_id=... direction=prompt
    header_re = re.compile(r"^\[(.*?)\] request_id=(.*?) direction=(.*?)$")
    
    with log_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
        
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = header_re.match(line)
        if match:
            timestamp, request_id, direction = match.groups()
            if direction == "prompt":
                # The next lines should be the JSON payload
                json_lines = []
                i += 1
                while i < len(lines):
                    json_line = lines[i]
                    # Check if next line is a header or empty line separator
                    if header_re.match(json_line.strip()) or json_line.strip() == "":
                        if json_line.strip() == "":
                             i += 1 # Skip empty line
                        break
                    json_lines.append(json_line)
                    i += 1
                
                json_str = "".join(json_lines)
                try:
                    payload = json.loads(json_str)
                    entries.append({
                        "timestamp": timestamp,
                        "request_id": request_id,
                        "payload": payload
                    })
                except json.JSONDecodeError:
                    print(f"Warning: Failed to parse JSON for request {request_id}", file=sys.stderr)
                    # continue outer loop from current i (which is at next header or empty line)
                    continue
                # We are already at the correct i for the next iteration (or one past)
                # continue to avoid incrementing i again at the end of loop if we used a while
                continue
        i += 1
        
    return entries

def diff_messages(msgs1: List[Dict], msgs2: List[Dict]) -> Dict[str, Any]:
    """
    Diffs two lists of messages.
    """
    # Helper to serialize message for comparison
    def serialize(msg):
        # We use a stable JSON representation for diffing
        return json.dumps(msg, sort_keys=True)

    seq1 = [serialize(m) for m in msgs1]
    seq2 = [serialize(m) for m in msgs2]
    
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    opcodes = matcher.get_opcodes()
    
    diffs = []
    stable_prefix_len = 0
    
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # This block is unchanged
            # If this is the first block (starts at 0), it contributes to stable prefix
            if i1 == 0 and j1 == 0:
                stable_prefix_len = i2 # i2 is exclusive, so length is i2 - 0
            
            diffs.append({
                "tag": "equal",
                "range_prev": [i1, i2],
                "range_curr": [j1, j2],
                "count": i2 - i1
            })
        elif tag == 'replace':
            diffs.append({
                "tag": "replace",
                "range_prev": [i1, i2],
                "range_curr": [j1, j2],
                "prev": msgs1[i1:i2],
                "curr": msgs2[j1:j2]
            })
        elif tag == 'delete':
            diffs.append({
                "tag": "delete",
                "range_prev": [i1, i2],
                "deleted": msgs1[i1:i2]
            })
        elif tag == 'insert':
            diffs.append({
                "tag": "insert",
                "range_curr": [j1, j2],
                "inserted": msgs2[j1:j2]
            })
            
    return {
        "stable_prefix_len": stable_prefix_len,
        "total_messages_prev": len(msgs1),
        "total_messages_curr": len(msgs2),
        "diffs": diffs
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 diff_turns.py <log_file> [turn_index_A] [turn_index_B]")
        sys.exit(1)
        
    log_file = Path(sys.argv[1])
    if not log_file.exists():
        print(f"Error: Log file {log_file} not found")
        sys.exit(1)
        
    entries = parse_log_file(log_file)
    if not entries:
        print("No prompt entries found in log file.")
        sys.exit(0)
        
    # Default to diffing the last two turns
    idx_a = -2
    idx_b = -1
    
    if len(sys.argv) >= 4:
        idx_a = int(sys.argv[2])
        idx_b = int(sys.argv[3])
    elif len(sys.argv) == 3:
        # If only one index provided, diff it against the previous one
        idx_b = int(sys.argv[2])
        idx_a = idx_b - 1
        
    # Handle negative indices
    if idx_a < 0: idx_a += len(entries)
    if idx_b < 0: idx_b += len(entries)
    
    if not (0 <= idx_a < len(entries)) or not (0 <= idx_b < len(entries)):
        print(f"Error: Indices {idx_a}, {idx_b} out of range (0-{len(entries)-1})")
        sys.exit(1)
        
    entry_a = entries[idx_a]
    entry_b = entries[idx_b]
    
    print(f"Diffing turn {idx_a} ({entry_a['timestamp']}) vs turn {idx_b} ({entry_b['timestamp']})")
    print(f"Request IDs: {entry_a['request_id']} -> {entry_b['request_id']}")
    
    msgs_a = entry_a['payload'].get('messages', [])
    msgs_b = entry_b['payload'].get('messages', [])
    
    result = diff_messages(msgs_a, msgs_b)
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
