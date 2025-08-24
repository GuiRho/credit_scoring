# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
# We need mlflow to parse the model, but also the specific libraries for the model to run
# (e.g., scikit-learn, xgboost). The requirements from the packaged model are perfect.
COPY production_model/requirements.txt model_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r model_requirements.txt

# Copy the application code and the packaged model
COPY docker_main.py .
COPY ../production_model ./model

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variable for MLflow tracking URI
# This is crucial for the API to fetch the run parameters
ENV MLFLOW_TRACKING_URI="file:///C:/Users/gui/Documents/OpenClassrooms/Projet%207/cache/mlruns"

# Run uvicorn server
# Use 0.0.0.0 to listen on all network interfaces
CMD ["uvicorn", "docker_main:app", "--host", "0.0.0.0", "--port", "8000"]