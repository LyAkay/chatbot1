FROM python:3.10-slim

# Cài đặt các công cụ hệ thống cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    swig \
    libopenblas-dev \
    libomp-dev

# Cập nhật pip
RUN pip install --no-cache-dir --upgrade pip

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép và cài đặt thư viện từ requirements.txt
COPY requirements.txt .
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
