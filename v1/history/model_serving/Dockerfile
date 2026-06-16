# Use a consistent Python version
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# COPY files from the build context (the project root) into the container
COPY model_serving/requirements.txt ./requirements.txt
COPY production_model/requirements.txt ./model_requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r model_requirements.txt

# COPY the application code and model from the project root
COPY model_serving/docker_main.py .
COPY production_model/ ./model/

# Expose the port Cloud Run uses
EXPOSE 8080

# The command to start the server
CMD ["uvicorn", "docker_main:app", "--host", "0.0.0.0", "--port", "8080"]