"""Watchdog for embedding_server.py — restarts it if it stops responding."""
import subprocess
import sys
import time
import urllib.request
import json
import signal
import os

HEALTH_URL = "http://localhost:8000/v1/embeddings"
HEALTH_BODY = json.dumps({"input": ["test"], "model": "all-MiniLM-L6-v2"}).encode()
CHECK_INTERVAL = 30   # seconds between health checks
FAIL_THRESHOLD = 2    # consecutive failures before restart
STARTUP_WAIT = 12     # seconds to wait after starting server

def health_check():
    try:
        req = urllib.request.Request(
            HEALTH_URL,
            data=HEALTH_BODY,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=10)
        return resp.status == 200
    except Exception:
        return False

def start_server():
    script = os.path.join(os.path.dirname(__file__), "embedding_server.py")
    proc = subprocess.Popen(
        [sys.executable, "-u", script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"[watchdog] Started embedding server PID={proc.pid}")
    time.sleep(STARTUP_WAIT)
    return proc

def kill_server(proc):
    if proc and proc.poll() is None:
        print(f"[watchdog] Killing server PID={proc.pid}")
        proc.kill()
        proc.wait(timeout=5)

def main():
    print("[watchdog] Embedding server watchdog started")
    proc = None
    consecutive_fails = 0

    try:
        while True:
            # Start server if not running
            if proc is None or proc.poll() is not None:
                if proc is not None:
                    print(f"[watchdog] Server exited with code {proc.returncode}")
                proc = start_server()
                consecutive_fails = 0

            # Health check
            if health_check():
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                print(f"[watchdog] Health check failed ({consecutive_fails}/{FAIL_THRESHOLD})")

                if consecutive_fails >= FAIL_THRESHOLD:
                    print("[watchdog] Restarting server...")
                    kill_server(proc)
                    proc = start_server()
                    consecutive_fails = 0

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("[watchdog] Shutting down...")
        kill_server(proc)

if __name__ == "__main__":
    main()
