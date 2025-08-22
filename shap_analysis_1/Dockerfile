FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shap_analysis_1.py .

CMD ["python", "shap_analysis_1.py"]