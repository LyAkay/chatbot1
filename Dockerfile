FROM python:3.10-slim

# Cài đặt các công cụ cần thiết để build thư viện
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    wget \
    swig \
    libopenblas-dev \
    libomp-dev \
    python3-dev

# Cập nhật pip và các công cụ build cần thiết
RUN pip install --no-cache-dir --upgrade pip setuptools wheel numpy

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép tệp yêu cầu và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn và mô hình vào container
COPY app/ /app/app/
COPY .env /app/.env
COPY app/models /app/models
COPY logs /app/logs
COPY pyproject.toml /app/pyproject.toml

# Thiết lập cổng mặc định
EXPOSE 8000

# Chạy ứng dụng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
