import os
import sys
import time
import socket
import webbrowser
import subprocess

APP_NAME = "NGX_AI_Optimization"
DEFAULT_PORT = int(os.environ.get("NGX_APP_PORT", "8501"))

def popup(msg: str):
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("NGX AI Optimization - Launch Error", msg)
    except Exception:
        pass

def port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False

def find_app_py(base_dir: str) -> str:
    meipass = getattr(sys, "_MEIPASS", None)
    candidates = [
        os.path.join(base_dir, "ngx_ai_app", "app.py"),
        os.path.join(base_dir, "_internal", "ngx_ai_app", "app.py"),
    ]
    if meipass:
        candidates.append(os.path.join(meipass, "ngx_ai_app", "app.py"))
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("app.py not found. Looked in:\n" + "\n".join(candidates))

def acquire_lock(base_dir: str) -> str:
    lock_path = os.path.join(base_dir, f".{APP_NAME}.lock")
    if os.path.exists(lock_path) and port_open(DEFAULT_PORT):
        webbrowser.open(f"http://localhost:{DEFAULT_PORT}")
        raise SystemExit(0)
    if os.path.exists(lock_path):
        try:
            os.remove(lock_path)
        except OSError:
            pass
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    return lock_path

def main():
    base_dir = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    log_path = os.path.join(base_dir, "launcher.log")
    lock = None
    try:
        lock = acquire_lock(base_dir)
        app_path = find_app_py(base_dir)

        if port_open(DEFAULT_PORT):
            webbrowser.open(f"http://localhost:{DEFAULT_PORT}")
            return

        cmd = [
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", str(DEFAULT_PORT),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
        ]

        with open(log_path, "a", encoding="utf-8") as log:
            log.write("\n\n=== Launching Streamlit ===\n")
            log.write("CMD: " + " ".join(cmd) + "\n")
            proc = subprocess.Popen(cmd, cwd=base_dir, stdout=log, stderr=log)

        start = time.time()
        opened = False
        while time.time() - start < 60:
            if proc.poll() is not None:
                raise RuntimeError("Streamlit exited immediately (startup failed). See launcher.log.")
            if port_open(DEFAULT_PORT):
                if not opened:
                    webbrowser.open(f"http://localhost:{DEFAULT_PORT}")
                    opened = True
                proc.wait()
                return
            time.sleep(0.25)

        raise TimeoutError("Streamlit did not become reachable in 60s. See launcher.log.")

    except Exception as e:
        popup(f"{e}\n\nCheck log:\n{log_path}")
        raise
    finally:
        if lock:
            try:
                os.remove(lock)
            except OSError:
                pass

if __name__ == "__main__":
    main()
