from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .rag import RAGPipeline
import logging

app = FastAPI(title="Chatbot API")
pipeline = RAGPipeline()

# Cấu hình logging
log_file = "logs/chatbot.log"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    try:
        response = pipeline.get_answer(query.text)
        return {"response": response}
    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
