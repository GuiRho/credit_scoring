# mlflow_utils.py

import os
import socket
import subprocess
import atexit
import time
from pathlib import Path
import mlflow

# --- Configuration ---
MLFLOW_HOST = "127.0.0.1"
MLFLOW_PORT = 5000
MLFLOW_TRACKING_URI = f"http://{MLFLOW_HOST}:{MLFLOW_PORT}"

# Global variable to hold the server process
_mlflow_server_process = None

def _is_server_running():
    """Checks if a service is running on the specified host and port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((MLFLOW_HOST, MLFLOW_PORT))
            return True
        except (ConnectionRefusedError, socket.timeout):
            return False

def _cleanup_mlflow_process():
    """Ensures the MLflow server process is terminated when the script exits."""
    global _mlflow_server_process
    if _mlflow_server_process and _mlflow_server_process.poll() is None:
        print("--- Shutting down managed MLflow server ---")
        _mlflow_server_process.terminate()
        _mlflow_server_process.wait()

def setup_mlflow(experiment_name: str, cache_dir: str):
    """
    The main setup function. It ensures the MLflow server is running and
    configures MLflow to connect to it.

    Args:
        experiment_name (str): The name of the MLflow experiment to use.
        cache_dir (str): The absolute path to your external cache directory.
    """
    global _mlflow_server_process

    if _is_server_running():
        print("--- MLflow server is already running. Connecting to it. ---")
    else:
        print("--- MLflow server not found. Starting a managed server... ---")
        
        # Define where the small DB and large artifacts will live
        project_root = Path(__file__).parent.resolve()
        backend_store_uri = f"sqlite:///{project_root / 'mlflow.db'}"
        artifact_root = Path(cache_dir) / "mlartifacts"
        
        # Ensure the artifact directory exists
        artifact_root.mkdir(parents=True, exist_ok=True)

        command = [
            "mlflow", "server",
            "--backend-store-uri", backend_store_uri,
            "--default-artifact-root", str(artifact_root),
            "--host", MLFLOW_HOST,
            "--port", str(MLFLOW_PORT),
        ]
        
        # Start the server as a background process
        _mlflow_server_process = subprocess.Popen(command)
        
        # Register the cleanup function to run when the script exits
        atexit.register(_cleanup_mlflow_process)
        
        # Give the server a moment to start up
        print("Waiting for MLflow server to start...")
        time.sleep(5)
        if not _is_server_running():
            raise RuntimeError("Failed to start the managed MLflow server.")
        print("--- MLflow server started successfully. ---")
        
    # Connect this script's MLflow client to the server
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow configured for experiment '{experiment_name}' at {MLFLOW_TRACKING_URI}")