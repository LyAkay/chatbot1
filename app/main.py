from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from app.services import RAGPipeline, log_response_to_csv
import logging

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
rag = RAGPipeline()

@app.post("/chat/")
async def chat(request: Request):
    try:
        # Đọc JSON từ body
        data = await request.json()
        query = data.get("query", "")

        # Kiểm tra nếu 'query' không có hoặc rỗng
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
