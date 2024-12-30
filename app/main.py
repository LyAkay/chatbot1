from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from app.services import RAGPipeline, log_response_to_csv
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng và pipeline
app = FastAPI()
rag = RAGPipeline()

@app.post("/chat/")
async def chat(request: Request):
    """
    Endpoint xử lý câu hỏi và trả về câu trả lời.
    """
    try:
        # Log toàn bộ body nhận được
        body = await request.body()
        logger.info(f"Received raw body: {body}")  # Ghi log nội dung thô
        logger.info(f"Decoded body: {body.decode('utf-8')}")  # Ghi log nội dung JSON

        # Chuyển body thành JSON
        data = await request.json()
        logger.info(f"Parsed JSON: {data}")  # Ghi log dữ liệu JSON đã phân tích

        # Kiểm tra nếu 'query' bị thiếu hoặc trống
        query = data.get("query", "")
        if not query:
            logger.error("Field 'query' is missing or empty.")
            raise HTTPException(
                status_code=422,
                detail="Field 'query' is required and cannot be empty."
            )

        # Xử lý câu hỏi
        answer = rag.get_answer(query)
        context = rag.get_relevant_context(query)

        # Lưu phản hồi vào CSV
        log_response_to_csv(query, answer, context)

        return {"query": query, "answer": answer}

    except Exception as e:
        # Log chi tiết lỗi
        logger.error(f"Error processing /chat/: {e}")
        return JSONResponse({"error": f"Internal server error: {str(e)}"}, status_code=500)

