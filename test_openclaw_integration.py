#!/usr/bin/env python3
import sys
import json
import time
import subprocess
import requests
import re
import os
import shutil
import signal
from pathlib import Path
from datetime import datetime

MC_URL = "http://localhost:3000/api"
LLM_URL = "http://127.0.0.1:4000/v1/models"
PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / f"logs/{datetime.now().strftime('%Y-%m-%d')}"
TEST_RESULTS_DIR = PROJECT_ROOT / "test_results"

class ServerManager:
    def __init__(self, run_id):
        self.run_id = run_id
        self.process = None
        self.run_dir = TEST_RESULTS_DIR / run_id
        self.server_log = self.run_dir / "server.log"
        self.pre_existing_logs = set()
        
    def setup(self):
        print(f"[*] Setting up test environment for run: {self.run_id}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Track existing logs to isolate new ones
        if LOG_DIR.exists():
            self.pre_existing_logs = set(LOG_DIR.glob("*.log"))
        
        # Kill any existing server
        self._kill_existing()
        
        # Start server
        print("[*] Starting MLX LLM server...")
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
        server_script = PROJECT_ROOT / "start-llm.py"
        
        if not venv_python.exists():
            print(f"[!] Virtual env not found at {venv_python}. Please run install_and_run.py first.")
            sys.exit(1)
            
        self.server_log_file = open(self.server_log, "w")
        self.process = subprocess.Popen(
            [str(venv_python), str(server_script)],
            stdout=self.server_log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT)
        )
        
        # Wait for ready
        self._wait_for_ready()
        
    def _kill_existing(self):
        print("[*] Cleaning up any existing LLM processes...")
        try:
            subprocess.run(["pkill", "-f", "start-llm.py"], stderr=subprocess.DEVNULL)
            subprocess.run(["pkill", "-f", "litellm"], stderr=subprocess.DEVNULL)
            time.sleep(2)
        except Exception:
            pass

    def _wait_for_ready(self, timeout=60):
        print("[*] Waiting for server to become ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                resp = requests.get(LLM_URL, timeout=2)
                if resp.status_code == 200:
                    print("[+] Server is ready.")
                    return
            except requests.RequestException:
                pass
            
            # Check if process crashed
            if self.process.poll() is not None:
                print(f"[!] Server process exited prematurely with code {self.process.returncode}")
                print(f"[!] Check {self.server_log} for details.")
                sys.exit(1)
                
            time.sleep(2)
            
        print("[!] Server failed to start within timeout.")
        self.teardown()
        sys.exit(1)
        
    def teardown(self):
        print(f"[*] Tearing down test environment...")
        if self.process:
            print("[*] Terminating server process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("[!] Server didn't terminate, killing...")
                self.process.kill()
        
        if hasattr(self, 'server_log_file') and not self.server_log_file.closed:
            self.server_log_file.close()
            
        # Collect new logs
        print(f"[*] Collecting session logs into {self.run_dir}")
        if LOG_DIR.exists():
            current_logs = set(LOG_DIR.glob("*.log"))
            new_logs = current_logs - self.pre_existing_logs
            for log in new_logs:
                shutil.copy2(log, self.run_dir / log.name)
                print(f"  - Copied {log.name}")
        
        self._kill_existing()


def run_openclaw_command(agent, message, session_id=None, timeout=None):
    cmd = ["openclaw", "agent", "--agent", agent, "--message", message, "--json"]
    if session_id:
        cmd.extend(["--session-id", session_id])
    
    print(f"[*] Running command for agent {agent}: {message}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            print(f"[!] Command failed: {result.stderr}")
            return None
        
        try:
            data = json.loads(result.stdout)
            return data
        except json.JSONDecodeError:
            print("[!] Failed to decode JSON from output")
            return None
    except subprocess.TimeoutExpired:
        print("[!] Command execution timed out.")
        return None

def check_mc_tasks():
    try:
        resp = requests.get(f"{MC_URL}/tasks")
        if resp.status_code == 200:
            return resp.json()
        return []
    except Exception as e:
        print(f"[!] MC API error: {e}")
        return []

def evaluate_server_log(run_dir):
    server_log = run_dir / "server.log"
    if not server_log.exists():
        print(f"[-] No server.log found in {run_dir}")
        return
        
    print(f"\n--- Server Debug Log Analysis ({run_dir.name}) ---")
    keywords = ["diverge", "evict", "flush", "delete", "dropping"]
    
    match_found = False
    with open(server_log, "r") as f:
        for line in f:
            if any(k in line.lower() for k in keywords):
                print(f"  > {line.strip()}")
                match_found = True
                
    if not match_found:
        print("  > No cache evictions, divergence, or flushes detected in server log.")

def evaluate_cache_logs(run_dir, session_id):
    logs = list(run_dir.glob("cache-session-*.log"))
    session_logs = []
    
    for logf in logs:
        with open(logf) as f:
            content = f.read()
            if session_id and session_id in content:
                session_logs.append(logf)
                
    if not session_logs:
        print(f"[-] No cache logs found containing session {session_id} in {run_dir}")
        return
        
    print(f"\n--- Cache Metrics for Session {session_id} ---")
    for logf in session_logs:
        with open(logf) as f:
            content = f.read()
            
        pattern = re.compile(
            r'\[[\d\-T:\.]+\] request_id=(\w+) direction=(\w+)\n(.*?)(?=\n\[[\d\-T:\.]+\] request_id=|\Z)',
            re.DOTALL
        )
        entries = pattern.findall(content)
        prompts = [(rid, body) for rid, d, body in entries if d == 'prompt']
        
        for i, (rid, body) in enumerate(prompts):
            try:
                d = json.loads(body)
                rm = d.get('request_meta', {})
                msgs = d.get('messages', [])
                pt = rm.get('prompt_tokens', 0)
                mpl = rm.get('matched_prefix_len', 0)
                cm = rm.get('cache_match_type', '?')
                hit = round(100 * mpl / pt, 1) if pt else 0
                spm = rm.get('stable_prefix_msg_count', 0)
                print(f"Turn {i}: {len(msgs)} msgs | Tokens: {pt} | Cache: {cm} ({hit}%) | Stable Msgs: {spm}")
            except Exception as e:
                pass

def test_scenario_1():
    run_id = f"S1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    server = ServerManager(run_id)
    
    try:
        server.setup()
        print("\n=== Starting Scenario 1: Simple Delegation ===")
        
        # Send initial command to Spock (may block during 5-minute prefill)
        msg = "Please write a simple python script that prints 'Hello from S1' and save it to /tmp/s1_test.py. Use dev_Dave to write it."
        
        # Since prefill can take ~5 mins, we don't put a tight timeout here.
        # It's bounded by Python's blocking subprocess.run by default.
        out = run_openclaw_command("pm_spock", msg)
        if not out:
            print("[!] Scenario 1 abort: Could not execute command.")
            return
            
        session_id = out.get('sessionId')
        print(f"[*] Started session: {session_id}")
        
        # Wait for processing and poll Mission Control (handle long LLM prefills)
        max_polls = 60  # 60 * 10s = 10 minutes (to allow 5m prefill + subagent runs)
        success = False
        for i in range(max_polls):
            print(f"[*] Polling Mission Control... ({i+1}/{max_polls})")
            time.sleep(10) # wait for LLM processing and MC updates
            tasks = check_mc_tasks()
            s1_tasks = [t for t in tasks if "s1_test.py" in t.get('details', '') or "s1_test.py" in t.get('title', '')]
            
            if s1_tasks:
                print(f"[*] Found {len(s1_tasks)} tasks related to S1 in Mission Control.")
                for t in s1_tasks:
                    print(f"  - Task {t['id']} | Status: {t['status']} | Progress: {t['progress']}% | Assignee: {t['assignee_id']}")
                    if t['status'] == 'completed':
                        success = True
            
            if success:
                print("[+] Scenario 1 Task Completed in Mission Control!")
                break

        if not success:
            print("[-] Scenario 1 did not complete within the polling window.")
            
    finally:
        server.teardown()
        
    # Evaluate Cache Logs
    if 'session_id' in locals() and session_id:
        evaluate_cache_logs(server.run_dir, session_id)
        
    # Evaluate Server Log
    evaluate_server_log(server.run_dir)
    
if __name__ == "__main__":
    test_scenario_1()
