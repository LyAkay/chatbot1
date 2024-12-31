import os
import logging
import json
import pickle
import numpy as np
import openai
import faiss
from dotenv import load_dotenv
from flask import Flask, request, jsonify

# Cấu hình logging
log_dir = "logs"
log_file = os.path.join(log_dir, "rag_pipeline.log")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Cấu hình thư mục cache và history
cache_file = "cache/cache_cache.json"
history_file = "history/chat_history.json"

os.makedirs("cache", exist_ok=True)
os.makedirs("history", exist_ok=True)

# Tạo tệp cache và history nếu chưa có
if not os.path.exists(cache_file):
    with open(cache_file, "w") as f:
        json.dump({}, f)

if not os.path.exists(history_file):
    with open(history_file, "w") as f:
        json.dump([], f)

# Load môi trường
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Khởi tạo FastAPI
app = FastAPI()

# Giới hạn lịch sử trò chuyện
MAX_HISTORY_LENGTH = 10

# Model quản lý
class ModelManager:
    def __init__(self):
        self.cache = self.load_cache()

    def load_cache(self):
        """Load cache từ file."""
        with open(cache_file, "r") as f:
            return json.load(f)

    def save_cache(self):
        """Lưu cache vào file."""
        with open(cache_file, "w") as f:
            json.dump(self.cache, f)

    def call_openai_model(self, prompt: str) -> str:
        """Gọi API OpenAI."""
        if prompt in self.cache:
            response = self.cache[prompt]
            logging.info(f"Retrieved answer from cache for prompt: {prompt}")
            return response

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500
            )
            response_content = response.choices[0].message["content"]
            self.cache[prompt] = response_content
            self.save_cache()
            return response_content
        except Exception as e:
            logging.error(f"OpenAI API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error from OpenAI API.")

# RAG Pipeline
class RAGPipeline:
    def __init__(self):
        try:
            self.index = faiss.read_index("faiss_index.bin")
            with open("chunked_texts.pkl", "rb") as f:
                self.texts = pickle.load(f)

            self.model_manager = ModelManager()

            self.prompt_template = """Trả lời câu hỏi một cách đầy đủ, súc tích và ngắn gọn bằng Tiếng Việt dựa trên (các) đoạn văn bản sau:
            {context}

            Câu hỏi: {question}

            Nếu không thể trả lời câu hỏi dựa trên ngữ cảnh, vui lòng trả lời một cách tự nhiên dưới dạng AI đàm thoại.
            """
            logging.info("RAG Pipeline initialized successfully")

        except Exception as e:
            logging.error(f"Error initializing RAG Pipeline: {str(e)}")
            raise HTTPException(status_code=500, detail="Error initializing RAG Pipeline.")

    def get_query_embedding(self, query: str) -> np.ndarray:
        """Tạo embedding cho query sử dụng OpenAI API."""
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-3-large"
        )
        return np.array(response["data"][0]["embedding"]).astype("float32")

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Lấy context liên quan từ chỉ mục."""
        query_embedding = self.get_query_embedding(query)
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return "\n\n".join([self.texts[i] for i in indices[0]])

    def get_answer(self, query: str) -> str:
        """Xử lý câu hỏi và trả về câu trả lời."""
        context = self.get_relevant_context(query)
        prompt = self.prompt_template.format(context=context, question=query)
        return self.model_manager.call_openai_model(prompt)

# Khởi tạo RAG pipeline
rag_pipeline = RAGPipeline()

# Endpoint FastAPI
@app.route('/answer', methods=['POST'])
def get_answer():
    data = request.get_json()
    query = data.get('query')
    selected_menu = data.get('selected_menu')
    chat_history = data.get('chat_history', [])

    if not query or not selected_menu:
        return jsonify({"error": "Missing 'query' or 'selected_menu' in request."}), 400

    vectors = faiss_indices.get(selected_menu)
    if not vectors:
        return jsonify({"error": f"No FAISS index found for menu '{selected_menu}'."}), 400

    if chat_history:
        previous_queries = ' '.join([f"User: {q}\nChatbot: {a}" for q, a in chat_history])
    else:
        previous_queries = ""

    # Check cache first
    cache_file = f"cache/{selected_menu.lower()}_cache.json"
    cached_answer = check_prompt_caching(query, cache_file)
    if cached_answer:
        logging.info(f"Returning cached answer for query: {query}")
        return jsonify({"answer": cached_answer})

    # Generate answer
    answer = generate_answer_with_rag(query, vectors, previous_queries, max_tokens=150)

    # Save to cache
    cache = load_json(cache_file)
    query_hash = hash_prompt(query)
    cache[query_hash] = answer
    save_json(cache_file, cache)

    response = {"answer": answer}
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
