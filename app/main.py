from fastapi import FastAPI
from app.services import RAGPipeline, log_response_to_csv

app = FastAPI()
rag = RAGPipeline()

@app.post("/chat/")
async def chat(query: str):
    """
    API xử lý câu hỏi và trả lời.
    """
    answer = rag.get_answer(query)
    context = rag.get_relevant_context(query)
    log_response_to_csv(query, answer, context)
    return {"query": query, "answer": answer}

@app.get("/download-responses/")
async def download_responses():
    """
    API để tải xuống file CSV chứa các phản hồi.
    """
    from fastapi.responses import FileResponse
    return FileResponse(path="app/responses.csv", filename="responses.csv", media_type="text/csv")
