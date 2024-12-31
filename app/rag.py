import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
import json
from datetime import datetime
from pathlib import Path
import csv
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("chatbot.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_file: str = "cache.json"):
        self.cache_file = Path(cache_file)
        self._load_cache()

    def _load_cache(self):
        try:
            with self.cache_file.open('r') as f:
                self.cache = json.load(f)
        except FileNotFoundError:
            self.cache = {}

    def save_cache(self):
        with self.cache_file.open('w') as f:
            json.dump(self.cache, f, indent=4)

    def get(self, query: str) -> Optional[str]:
        return self.cache.get(query.lower().strip())

    def add(self, query: str, response: str):
        self.cache[query.lower().strip()] = {
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        self.save_cache()

class RAGSystem:
    def __init__(self,
                 index_file: str = "app/models/faiss_index.bin",
                 texts_file: str = "app/models/chunked_texts.pkl",
                 cache_file: str = "cache.json"):
        self.index_file = Path(index_file)
        self.texts_file = Path(texts_file)
        self.cache_file = Path(cache_file)
        self.client = self._init_openai_client()
        self.cache = CacheManager(cache_file=str(self.cache_file))
        self._load_resources()

    def _init_openai_client(self) -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        return OpenAI(api_key=api_key)

    def _load_resources(self):
        try:
            self.index = faiss.read_index(str(self.index_file))
            with self.texts_file.open('rb') as f:
                self.texts = pickle.load(f)
        except FileNotFoundError as e:
            raise Exception(f"Resource files not found: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        if not text:
            raise ValueError("Input text cannot be empty.")

        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding, dtype='float32')

    def get_context(self, query: str, k: int = 3) -> str:
        query_vector = self.get_embedding(query)
        D, I = self.index.search(query_vector.reshape(1, -1), k)
        contexts = [self.texts[i] for i in I[0] if i < len(self.texts)]
        return "\n\n".join(dict.fromkeys(contexts))

    def get_answer(self, query: str) -> Dict[str, str | float]:
        start_time = datetime.now()

        cached = self.cache.get(query)
        if cached:
            response_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Response from cache: {query} - {cached['response']} - {response_time:.2f}s")
            return {
                'response': cached['response'],
                'source': 'cache',
                'response_time': response_time
            }

        try:
            context = self.get_context(query)
            prompt = f"""Provide a comprehensive, concise, and brief answer in Vietnamese based on the following text:
            {context}

            Question: {query}

            If the question cannot be answered based on the context, please reply: "Tôi không tìm thấy thông tin về vấn đề này trong dữ liệu."
            """

            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            answer = response.choices[0].message.content
            response_time = (datetime.now() - start_time).total_seconds()

            self.cache.add(query, answer)
            with open('responses.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([query, answer, response_time])
            logger.info(f"Response from API: {query} - {answer} - {response_time:.2f}s")

            return {
                'response': answer,
                'source': 'api',
                'response_time': response_time
            }

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Error generating response: {e} - {response_time:.2f}s")
            raise
