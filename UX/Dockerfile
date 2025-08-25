# Use a consistent Python version
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# COPY files from the build context (the project root) into the container
COPY ux/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# COPY the app code from the project root into the container
COPY ux/app.py .

# Expose the port Cloud Run uses
EXPOSE 8080

# The command to start the Streamlit server
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]