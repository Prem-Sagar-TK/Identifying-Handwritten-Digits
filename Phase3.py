import subprocess
import sys
import os
import time
import threading

def log_stream(stream, prefix):
    try:
        for line in iter(stream.readline, ""):
            if line:
                print(f"[{prefix}] {line.strip()}")
    except Exception as e:
        print(f"[{prefix}] Stream closed: {e}")

def main():
    print("=" * 60)
    print("Starting Digit Recognizer Web App (Flask Backend + React Frontend)...")
    print("=" * 60)

    # 1. Start Flask backend
    # Run using the same python interpreter
    backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
    backend_cmd = [sys.executable, "-u", "backend_app.py"]
    print(f"Starting Flask backend in {backend_dir}: {' '.join(backend_cmd)}")
    
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=backend_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Allow backend a moment to boot
    time.sleep(1.5)

    # 2. Start Vite Frontend (needs shell=True on Windows to resolve npm cmd wrapper)
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
    print(f"Starting Vite frontend in: {frontend_dir}")
    
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=frontend_dir,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # 3. Create threads to print stdout/stderr streams with nice prefixes
    t_back_out = threading.Thread(target=log_stream, args=(backend_proc.stdout, "BACKEND"), daemon=True)
    t_back_err = threading.Thread(target=log_stream, args=(backend_proc.stderr, "BACKEND-ERR"), daemon=True)
    t_front_out = threading.Thread(target=log_stream, args=(frontend_proc.stdout, "FRONTEND"), daemon=True)
    t_front_err = threading.Thread(target=log_stream, args=(frontend_proc.stderr, "FRONTEND-ERR"), daemon=True)

    t_back_out.start()
    t_back_err.start()
    t_front_out.start()
    t_front_err.start()

    print("\nApplication is booting...")
    print("  - Backend health check: http://localhost:5000/health")
    print("  - Frontend server:       http://localhost:5173  (typically)")
    print("\nPress Ctrl+C to terminate both servers.")
    print("-" * 60)

    try:
        while True:
            # Check if any process terminated
            if backend_proc.poll() is not None:
                print(f"\n[SYSTEM] Backend process terminated with exit code {backend_proc.poll()}")
                break
            if frontend_proc.poll() is not None:
                print(f"\n[SYSTEM] Frontend process terminated with exit code {frontend_proc.poll()}")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Ctrl+C detected. Shutting down servers...")
    finally:
        # Gracefully terminate
        try:
            backend_proc.terminate()
            backend_proc.wait(timeout=2)
        except Exception:
            backend_proc.kill()

        try:
            frontend_proc.terminate()
            frontend_proc.wait(timeout=2)
        except Exception:
            frontend_proc.kill()
            
        print("[SYSTEM] Shutdown complete.")

if __name__ == "__main__":
    main()
