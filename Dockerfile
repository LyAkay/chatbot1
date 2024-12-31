FROM python:3.10-slim

# Tạo user mới
RUN useradd -m myuser
USER myuser
WORKDIR /home/myuser/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Thêm /home/myuser/.local/bin vào PATH
ENV PATH="/home/myuser/.local/bin:${PATH}"

COPY app/ /home/myuser/app/app/
COPY .env /home/myuser/app/.env
COPY app/models /home/myuser/app/models

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
