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
        try:
            self.index = faiss.read_index('faiss_index.bin')
            with open('chunked_texts.pkl', 'rb') as f:
                self.texts = pickle.load(f)

            self.model_manager = ModelManager()
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

            self.prompt_template = """"Trả lời câu hỏi một cách đầy đủ, súc tích và ngắn gọn bằng Tiếng Việt dựa trên (các) đoạn văn bản sau:
            {context}

            Câu hỏi: {question}

            Nếu không thể trả lời câu hỏi dựa trên ngữ cảnh, vui lòng trả lời một cách tự nhiên dưới dạng AI đàm thoại. Vui lòng trả lời tất cả các câu hỏi bằng Tiếng Việt.

            """
            self.chat_history = deque(maxlen=MAX_HISTORY_LENGTH)
            self.full_chat_history = [] # Lưu trữ toàn bộ lịch sử hội thoại
            logging.info("RAG Pipeline initialized successfully")

        except Exception as e:
            logging.error(f"Error initializing RAG Pipeline: {str(e)}")
            raise

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Tạo embedding cho query sử dụng OpenAI API."""
        if not self.model_manager.openai_client:
            raise Exception("OpenAI API key not found.")
        try:
            response = self.model_manager.openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-large"
            )
            return np.array(response.data[0].embedding).astype('float32')
        except Exception as e:
            logging.error(f"Error creating query embedding: {str(e)}")
            raise

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Lấy context liên quan từ index."""
        try:
            query_embedding = self.get_query_embedding(query)
            D, I = self.index.search(np.array([query_embedding]), k)
            contexts = [self.texts[i] for i in I[0]]
            unique_contexts = []
            seen_contexts = set()
            for c in contexts:
                if c not in seen_contexts:
                    unique_contexts.append(c)
                    seen_contexts.add(c)
            context = "\n\n".join(unique_contexts)
            return context
        except Exception as e:
            logging.error(f"Error retrieving context: {str(e)}")
            raise

    def get_answer(self, query: str) -> str:
        """Xử lý câu hỏi và trả về câu trả lời."""
        try:
            # Lấy ngữ cảnh liên quan
            context = self.get_relevant_context(query)

            # Không đưa lịch sử trò chuyện vào prompt
            prompt = self.prompt_template.format(context=context, question=query)

            # Sử dụng model OpenAI
            answer = self.model_manager.call_openai_model(prompt)
            self.update_chat_history(query, answer)
            return answer

        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return f"Lỗi: {str(e)}"

    def update_chat_history(self, query: str, answer: str):
        """Cập nhật lịch sử trò chuyện."""
        self.chat_history.append({"sender": "User", "text": query})
        self.chat_history.append({"sender": "Bot", "text": answer})
        # Lưu trữ riêng toàn bộ lịch sử
        self.full_chat_history.append({"sender": "User", "text": query})
        self.full_chat_history.append({"sender": "Bot", "text": answer})

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
