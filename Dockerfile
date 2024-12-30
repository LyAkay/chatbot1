FROM python:3.9-slim

# Đặt thư mục làm việc
WORKDIR /app

# Copy các tệp vào container
COPY . /app

# Cài đặt thư viện
RUN pip install --no-cache-dir -r requirements.txt

# Chạy ứng dụng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
