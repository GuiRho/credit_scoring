# FIX 1: Use a Python version that matches your model's environment
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files
COPY model_serving/requirements.txt ./requirements.txt
COPY production_model/requirements.txt ./model_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt -r model_requirements.txt

# Copy the application code
COPY model_serving/docker_main.py .

# FIX 2: Correctly copy the entire model directory to a folder named "model"
COPY production_model/ ./model/

# Expose the port Cloud Run will use
EXPOSE 8080

# Run the application (use an array to avoid shell issues, PORT is set by Cloud Run)
CMD ["uvicorn", "docker_main:app", "--host", "0.0.0.0", "--port", "8080"]