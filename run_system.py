
import os
import sys
import subprocess
import time
import webbrowser

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

EYE_MONITOR = os.path.join(BASE_DIR, "eye_monitor.py")
DASHBOARD = os.path.join(BASE_DIR, "dashboard.py")

eye_process = None
dashboard_process = None


def start_system():

    global eye_process, dashboard_process

    print("=" * 60)
    print("       AI DRIVER MONITORING SYSTEM")
    print("=" * 60)

    # Start eye monitoring
    print("\n[INFO] Starting Eye Monitoring Engine...")

    eye_process = subprocess.Popen(
        [PYTHON, "-u", EYE_MONITOR],
        cwd=BASE_DIR
    )

    time.sleep(3)

    if eye_process.poll() is not None:

        print("[ERROR] Eye monitoring stopped during startup!")
        print("Exit code:", eye_process.returncode)

        return False

    print("[SUCCESS] Eye monitoring is running")

    # Start Streamlit correctly
    print("\n[INFO] Starting Streamlit Dashboard...")

    dashboard_process = subprocess.Popen(
        [
            PYTHON,
            "-m",
            "streamlit",
            "run",
            DASHBOARD,
            "--server.port", "8501",
            "--server.address", "127.0.0.1",
            "--server.headless", "true"
        ],
        cwd=BASE_DIR
    )

    time.sleep(5)

    if dashboard_process.poll() is not None:

        print("[ERROR] Dashboard stopped!")
        print("Exit code:", dashboard_process.returncode)

        return False

    webbrowser.open("http://localhost:8501")

    print("\n" + "=" * 60)
    print("SYSTEM RUNNING")
    print("=" * 60)

    print("AI Monitoring : RUNNING")
    print("Dashboard     : RUNNING")
    print("URL           : http://localhost:8501")

    return True


def monitor_system():

    while True:

        if eye_process.poll() is not None:

            print("\n[ERROR] Eye monitoring process crashed!")

            print("Exit code:", eye_process.returncode)

            break

        if dashboard_process.poll() is not None:

            print("\n[ERROR] Dashboard process crashed!")

            print("Exit code:", dashboard_process.returncode)

            break

        time.sleep(2)


def stop_system():

    print("\n[INFO] Stopping system...")

    for process in [eye_process, dashboard_process]:

        if process is not None and process.poll() is None:

            process.terminate()

            try:
                process.wait(timeout=5)

            except subprocess.TimeoutExpired:
                process.kill()

    print("[INFO] System stopped")


if __name__ == "__main__":

    try:

        if start_system():
            monitor_system()

    except KeyboardInterrupt:

        print("\n[INFO] Ctrl+C received")

    finally:

        stop_system()
