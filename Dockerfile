FROM python:3.11-slim

WORKDIR /app

# libgomp1 is needed for XGBoost's OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

# Run from /app so relative paths in api/main.py (../models) resolve correctly
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
