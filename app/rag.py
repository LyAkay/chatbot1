import os
import logging
import pickle
import numpy as np
import faiss
from openai import OpenAI
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Cấu hình logging
log_file = "logs/chatbot.log"
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class CacheManager:
    def __init__(self, cache_file: str = "cache.json"):
        self.cache_file = Path(cache_file)
        self._load_cache()

    def _load_cache(self):
        """Tải cache từ file."""
        try:
            if self.cache_file.exists():
                with self.cache_file.open("r") as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except Exception as e:
            logging.error(f"Error loading cache: {str(e)}")
            self.cache = {}

    def save_cache(self):
        """Lưu cache vào file."""
        try:
            with self.cache_file.open("w") as f:
                json.dump(self.cache, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving cache: {str(e)}")

    def get(self, query: str):
        """Lấy câu trả lời từ cache."""
        return self.cache.get(query.lower().strip())

    def add(self, query: str, response: str):
        """Thêm câu trả lời mới vào cache."""
        self.cache[query.lower().strip()] = {
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.save_cache()

class RAGPipeline:
    def __init__(self):
        # Tải biến môi trường
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
        
        self.client = self._init_openai_client()
        self.cache = CacheManager()
        self.index = self._load_faiss_index("app/models/faiss_index.bin")
        self.texts = self._load_chunked_texts("app/models/chunked_texts.pkl")

        self.prompt_template = """
        Trả lời câu hỏi một cách đầy đủ, súc tích và ngắn gọn bằng Tiếng Việt dựa trên đoạn văn bản sau:
        {context}

        Câu hỏi: {question}

        Nếu không thể trả lời câu hỏi dựa trên ngữ cảnh, vui lòng trả lời: "Tôi không tìm thấy thông tin về vấn đề này trong dữ liệu."
        """
        logging.info("RAGPipeline initialized successfully.")

    def _init_openai_client(self):
        """Khởi tạo OpenAI client."""
        try:
            return OpenAI(api_key=self.api_key)
        except Exception as e:
            logging.error(f"Error initializing OpenAI client: {str(e)}")
            raise

    def _load_faiss_index(self, index_file: str):
        """Tải FAISS index từ file."""
        try:
            return faiss.read_index(index_file)
        except Exception as e:
            logging.error(f"Error loading FAISS index: {str(e)}")
            raise

    def _load_chunked_texts(self, texts_file: str):
        """Tải các đoạn văn bản chunked từ file."""
        try:
            with open(texts_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading chunked texts: {str(e)}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """Tạo embedding cho đoạn văn bản."""
        if not text:
            raise ValueError("Input text cannot be empty.")
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return np.array(response.data[0].embedding).astype("float32")
        except Exception as e:
            logging.error(f"Error generating embedding: {str(e)}")
            raise

    def get_context(self, query: str, k: int = 3):
        """Lấy các đoạn văn bản liên quan từ FAISS index."""
        try:
            query_embedding = self.get_embedding(query)
            distances, indices = self.index.search(np.array([query_embedding]), k)
            contexts = [self.texts[i] for i in indices[0] if i < len(self.texts)]
            return "\n\n".join(contexts)
        except Exception as e:
            logging.error(f"Error retrieving context: {str(e)}")
            raise

    def get_answer(self, query: str) -> dict:
        """Tạo câu trả lời cho câu hỏi dựa trên ngữ cảnh và cache."""
        try:
            # Kiểm tra cache
            cached_response = self.cache.get(query)
            if cached_response:
                logging.info(f"Cache hit for query: {query}")
                return cached_response["response"]

            # Lấy ngữ cảnh
            context = self.get_context(query)

            # Chuẩn bị prompt
            prompt = self.prompt_template.format(context=context, question=query)

            # Gọi OpenAI API để lấy câu trả lời
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500
            )

            answer = response.choices[0].message.content
            self.cache.add(query, answer)
            logging.info(f"Generated answer for query: {query}")
            return answer
        except Exception as e:
            logging.error(f"Error generating answer: {str(e)}")
            raise
