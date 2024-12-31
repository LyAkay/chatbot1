FROM python:3.10

# Thiết lập thư mục làm việc
WORKDIR /app

# Cập nhật pip và cài đặt các thư viện cần thiết
RUN pip install --no-cache-dir --upgrade pip

# Sao chép tệp yêu cầu và cài đặt thư viện
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
