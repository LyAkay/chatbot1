FROM python:3.10-slim

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép file yêu cầu và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép mã nguồn vào container
COPY app/ /app/app/
COPY .env /app/.env
COPY app/models /app/models

# Khởi chạy ứng dụng
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
