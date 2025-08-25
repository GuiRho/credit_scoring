# Use a consistent Python version
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# COPY files using the correct case-sensitive path
COPY UX/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# COPY the app code using the correct case-sensitive path
COPY UX/app.py .

# Expose the port Cloud Run uses
EXPOSE 8080

# The command to start the Streamlit server
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]