FROM python:3.10-slim

# Cài đặt các công cụ cần thiết
RUN apt-get update && apt-get install -y \
    build-essential \
    swig \
    wget \
    libopenblas-dev \
    libomp-dev \
    python3-dev

# Cập nhật pip và cài đặt numpy, setuptools, wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel numpy

# Cài đặt faiss-cpu từ pre-built wheel
RUN pip install --no-cache-dir https://github.com/facebookresearch/faiss/releases/download/v1.7.2/faiss_cpu-1.7.2-cp310-cp310-manylinux2014_x86_64.whl

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
