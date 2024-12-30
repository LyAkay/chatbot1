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

@app.get("/")
async def root():
    """
    Endpoint gốc để kiểm tra ứng dụng hoạt động.
    """
    return {"message": "Ứng dụng Chatbot hoạt động! Truy cập /chat/ để gửi câu hỏi."}

@app.post("/chat/")
async def chat(request: Request):
    """
    Endpoint xử lý câu hỏi và trả về câu trả lời.
    """
    try:
        # Log toàn bộ body nhận được
        body = await request.body()
        try:
            decoded_body = body.decode('utf-8')
            logger.info(f"Received raw body: {decoded_body}")  # Ghi log nội dung JSON
        except Exception as decode_error:
            logger.error(f"Error decoding body: {decode_error}")
            raise HTTPException(
                status_code=400,
                detail="Invalid encoding. Ensure the request body is UTF-8 encoded."
            )

        # Chuyển body thành JSON
        try:
            data = await request.json()
            logger.info(f"Parsed JSON: {data}")  # Ghi log dữ liệu JSON đã phân tích
        except Exception as json_error:
            logger.error(f"Error parsing JSON: {json_error}")
            raise HTTPException(
                status_code=422,
                detail=f"Invalid JSON format: {json_error}"
            )

        # Kiểm tra nếu 'query' bị thiếu hoặc rỗng
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

    except HTTPException as he:
        # Log chi tiết lỗi HTTP
        logger.error(f"HTTPException: {he.detail}")
        raise he
    except Exception as e:
        # Log chi tiết lỗi
        logger.error(f"Error processing /chat/: {e}")
        return JSONResponse({"error": f"Internal server error: {str(e)}"}, status_code=500)

@app.get("/download-responses/")
async def download_responses():
    """
    Endpoint để tải file CSV chứa các phản hồi.
    """
    try:
        return FileResponse(path="app/responses.csv", filename="responses.csv", media_type="text/csv")
    except Exception as e:
        logger.error(f"Error processing /download-responses/: {e}")
        return JSONResponse({"error": f"Internal server error: {str(e)}"}, status_code=500)
