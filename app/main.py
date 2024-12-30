import os
import csv
import time
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
import openai

# Tải biến môi trường
load_dotenv()

# Đường dẫn file
FAISS_INDEX = "faiss_index.bin"
TEXTS_FILE = "chunked_texts.pkl"
CSV_FILE = "app/responses.csv"

class RAGPipeline:
    def __init__(self):
        self.index = faiss.read_index(FAISS_INDEX)
        with open(TEXTS_FILE, "rb") as f:
            self.texts = pickle.load(f)

    def get_query_embedding(self, query: str) -> np.ndarray:
        """
        Tạo embedding cho câu hỏi bằng API OpenAI.
        """
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Embedding.create(
            model="text-embedding-3-large",  # Mô hình Embedding chính thức từ OpenAI
            input=query
        )
        return np.array(response["data"][0]["embedding"], dtype="float32")

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """
        Lấy ngữ cảnh liên quan từ FAISS index.
        """
        query_embedding = self.get_query_embedding(query)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        contexts = [self.texts[idx] for idx in indices[0]]
        return "\n\n".join(contexts)

    def get_answer(self, query: str) -> str:
        """
        Gửi câu hỏi đến OpenAI để nhận câu trả lời.
        """
        openai.api_key = os.getenv("OPENAI_API_KEY")
        context = self.get_relevant_context(query)
        prompt = f"""
        Trả lời câu hỏi sau dựa trên ngữ cảnh:
        {context}

        Câu hỏi: {query}
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Hoặc "gpt-4" nếu cần
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()

def log_response_to_csv(question: str, answer: str, context: str, reviewed=False, rating=None, comments=None):
    """
    Lưu câu hỏi, câu trả lời và ngữ cảnh vào file CSV.
    """
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Question", "Answer", "Context", "Timestamp", "Reviewed", "Rating", "Comments"])

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([None, question, answer, context, time.strftime("%Y-%m-%d %H:%M:%S"), reviewed, rating, comments])
