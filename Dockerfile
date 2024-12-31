FROM python:3.10-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/
COPY .env /app/.env
COPY app/models /app/models

# Multi-stage build: copy only necessary files to a new image
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
