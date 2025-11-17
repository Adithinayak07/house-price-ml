import subprocess
import time
import webbrowser
import os
import signal

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PYTHON = os.path.join(PROJECT_DIR, "mlvenv", "Scripts", "python.exe")


def start_process(command_list, wait=5):
    print(f"\n🚀 Starting: {' '.join(command_list)}\n")
    process = subprocess.Popen(
        command_list,
        cwd=PROJECT_DIR,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
    )
    time.sleep(wait)
    return process


def main():
    print("===================================")
    print("   🏡 House Rent ML Pipeline Run")
    print("===================================")

    # -------------------------------
    # 1. Start MLflow server
    # -------------------------------
    mlflow_cmd = [
        ENV_PYTHON, "-m", "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--default-artifact-root", "./mlruns",
        "--port", "5000"
    ]
    mlflow_proc = start_process(mlflow_cmd, wait=6)
    print("MLflow UI → http://localhost:5000")

    # -------------------------------
    # 2. Train the model
    # -------------------------------
    train_cmd = [
        ENV_PYTHON, "-m", "src.models.train",
        "--config", "config/config.yaml"
    ]
    subprocess.run(train_cmd, cwd=PROJECT_DIR)
    print("\n✔ Model training completed.\n")

    # -------------------------------
    # 3. Start FastAPI server
    # -------------------------------
    fastapi_cmd = [
        ENV_PYTHON, "-m", "uvicorn", "src.api.fastapi_app:app",
        "--port", "8000", "--reload"
    ]
    fastapi_proc = start_process(fastapi_cmd)
    print("FastAPI → http://localhost:8000")

    # -------------------------------
    # 4. Start Streamlit UI
    # -------------------------------
    streamlit_cmd = [
        ENV_PYTHON, "-m", "streamlit", "run", "src/ui/app.py"
    ]
    streamlit_proc = start_process(streamlit_cmd, wait=6)
    print("Streamlit → http://localhost:8501")

    webbrowser.open("http://localhost:8501")

    print("\n🚀 All services are running!")
    print("Press CTRL+C to stop everything.\n")

    # -------------------------------
    # Keep script alive
    # -------------------------------
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")

        for p in [mlflow_proc, fastapi_proc, streamlit_proc]:
            try:
                os.kill(p.pid, signal.SIGTERM)
            except:
                pass

        print("✔ All services stopped.")


if __name__ == "__main__":
    main()
