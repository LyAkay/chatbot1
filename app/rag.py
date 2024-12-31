import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from typing import Dict, Optional
import json
from datetime import datetime

class CacheManager:
    def __init__(self, cache_file: str = "cache.json"):
        self.cache_file = cache_file
        self._load_cache()
    
    def _load_cache(self):
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            self.cache = {}
    
    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
    
    def get(self, query: str) -> Optional[str]:
        return self.cache.get(query.lower().strip())
    
    def add(self, query: str, response: str):
        self.cache[query.lower().strip()] = {
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        self.save_cache()

class RAGSystem:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cache = CacheManager()
        self._load_resources()

    def _load_resources(self):
        try:
            self.index = faiss.read_index('models/faiss_index.bin')
            with open('models/chunked_texts.pkl', 'rb') as f:
                self.texts = pickle.load(f)
        except FileNotFoundError as e:
            raise Exception(f"Resource files not found: {str(e)}")

    def get_embedding(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return np.array(response.data[0].embedding, dtype='float32')

    def get_context(self, query: str, k: int = 3) -> str:
        query_vector = self.get_embedding(query)
        D, I = self.index.search(query_vector.reshape(1, -1), k)
        contexts = [self.texts[i] for i in I[0] if i < len(self.texts)]
        return "\n\n".join(dict.fromkeys(contexts))

    def get_answer(self, query: str) -> Dict:
        # Kiểm tra cache
        cached = self.cache.get(query)
        if cached:
            return {
                'response': cached['response'],
                'source': 'cache'
            }

        try:
            context = self.get_context(query)
            prompt = f"""Trả lời câu hỏi một cách đầy đủ, súc tích và ngắn gọn bằng Tiếng Việt dựa trên đoạn văn bản sau:
            {context}

            Câu hỏi: {query}

            Nếu không thể trả lời câu hỏi dựa trên ngữ cảnh, vui lòng trả lời: "Tôi không tìm thấy thông tin về vấn đề này trong dữ liệu."
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            answer = response.choices[0].message.content
            self.cache.add(query, answer)
            
            return {
                'response': answer,
                'source': 'api'
            }

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")
