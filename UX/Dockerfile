# Use a consistent Python version
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy necessary directories
COPY model_analysis/ ./model_analysis/
COPY production_model/ ./production_model/
COPY UX/ ./UX/
COPY test_processed.parquet ./

# Install dependencies from both requirements files
COPY model_analysis/requirements.txt ./model_analysis_requirements.txt
RUN pip install --no-cache-dir -r model_analysis_requirements.txt

COPY UX/requirements.txt ./ux_requirements.txt
RUN pip install --no-cache-dir -r ux_requirements.txt

# Run the analysis script to generate data for the dashboard
RUN python model_analysis/new_analysis.py

# Expose the port Cloud Run uses
EXPOSE 8080

# The command to start the Streamlit server with the new app
CMD ["streamlit", "run", "UX/new_app.py", "--server.port=8080", "--server.address=0.0.0.0"]