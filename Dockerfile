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
COPY streamlit_app.py .

EXPOSE 8000

# Default command runs the FastAPI service. For a second Railway service running
# Streamlit from this same image, override the start command in Railway's settings:
#   streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
