# chatbotcathe
# RAG Pipeline Chatbot

## Mô tả
Dự án xây dựng hệ thống chatbot dựa trên FAISS và OpenAI.

## Yêu cầu
- Python 3.9+
- Tài khoản OpenAI API
- FAISS index (`faiss_index.bin`) và file chunked texts (`chunked_texts.pkl`)

## Cài đặt
1. Cài đặt thư viện:
    ```bash
    pip install -r requirements.txt
    ```
2. Thiết lập biến môi trường trong file `.env`:
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    ```
3. Chạy ứng dụng:
    ```bash
    uvicorn app.main:app --reload
    ```

## API Endpoints
- `POST /chat/`: Gửi câu hỏi và nhận câu trả lời.
- `GET /download-responses/`: Tải file CSV chứa các phản hồi.

## Triển khai bằng Docker
1. Xây dựng container:
    ```bash
    docker build -t rag-chatbot .
    ```
2. Chạy container:
    ```bash
    docker run -p 8000:8000 rag-chatbot
    ```
