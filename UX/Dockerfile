# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit application code
COPY app.py .

# Make port 8501 available (Streamlit's default port)
EXPOSE 8501

# Command to run the Streamlit app
# We will set the API_URL via an environment variable in Cloud Run
CMD ["streamlit", "run", "app.py"]