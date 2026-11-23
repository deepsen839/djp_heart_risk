FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY app ./app
RUN pip install --upgrade pip && pip install ".[ui,mlops,features]"
RUN mkdir -p /app/models /app/reports /app/data /app/configs
EXPOSE 8000 8501
CMD ["uvicorn", "djp_heart_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
