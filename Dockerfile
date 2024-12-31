FROM python:3.10-slim AS builder

# Tạo user mới
RUN useradd -m myuser
USER myuser
WORKDIR /home/myuser/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /home/myuser/app/app/
COPY .env /home/myuser/app/.env
COPY app/models /home/myuser/app/models

# Thêm dòng này để cập nhật PATH
ENV PATH="/home/myuser/.local/bin:${PATH}"

# Multi-stage build: copy only necessary files to a new image
FROM python:3.10-slim

# Tạo user mới trong image cuối cùng
RUN useradd -m myuser
USER myuser
WORKDIR /home/myuser/app

COPY --from=builder /home/myuser/app /home/myuser/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
