from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag import RAGSystem
from typing import Dict

app = FastAPI(title="Chatbot API")
rag_system = RAGSystem()

class Query(BaseModel):
    text: str

class Response(BaseModel):
    response: str
    source: str
    response_time: float

@app.post("/chat", response_model=Response)
async def chat(query: Query) -> dict:
    try:
        result = rag_system.get_answer(query.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

@app.get("/health")
async def health_check() -> dict:
    return {"status": "healthy"}
