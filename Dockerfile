# Sử dụng Python slim để giảm kích thước container
FROM python:3.9-slim

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép toàn bộ mã nguồn vào container
COPY . /app

# Cài đặt thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Chạy ứng dụng với Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
