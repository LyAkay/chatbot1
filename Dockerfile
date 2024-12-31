FROM python:3.10-slim

# Cài đặt công cụ cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    wget \
    libopenblas-dev \
    libomp-dev

# Cập nhật pip và cài đặt setuptools, wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép tệp yêu cầu và cài đặt các thư viện
COPY requirements.txt .
COPY pyproject.toml .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn và mô hình vào container
COPY app/ /app/app/
COPY .env /app/.env
COPY app/models /app/models
COPY logs /app/logs

# Thiết lập cổng mặc định
EXPOSE 8000

# Chạy ứng dụng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
