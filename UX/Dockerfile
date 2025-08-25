# Use a consistent Python version
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# The paths are now simple because the build will happen from within the 'ux' folder
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Expose the port Cloud Run uses
EXPOSE 8080

# The command to start the Streamlit server
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]