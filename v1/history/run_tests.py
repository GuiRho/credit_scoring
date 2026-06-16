import subprocess
import sys

if __name__ == "__main__":
    # First, uninstall both mlflow and mlflow-skinny to ensure a clean state
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "mlflow", "mlflow-skinny", "-y"], capture_output=True)
    
    # Now, install only mlflow-skinny
    subprocess.run([sys.executable, "-m", "pip", "install", "mlflow-skinny<2.12"], capture_output=True)
    
    # Run pytest
    result = subprocess.run([sys.executable, "-m", "pytest"], capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr)
    sys.exit(result.returncode)