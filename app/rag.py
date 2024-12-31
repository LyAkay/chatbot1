import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from typing import Dict, Optional
import json
import csv
from datetime import datetime

class ResponseLogger:
    def __init__(self, log_file: str = "responses.csv"):
        self.log_file = log_file
        self._init_log_file()
    
    def _init_log_file(self):
        # Tạo file nếu chưa tồn tại
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'query', 'response', 'source', 'response_time'])
    
    def log_response(self, query: str, response: str, source: str, response_time: float):
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                query,
                response,
                source,
                f"{response_time:.2f}"
            ])

class RAGSystem:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.cache = CacheManager()
        self.logger = ResponseLogger()
        self._load_resources()

    # ... (các phương thức khác giữ nguyên)

    def get_answer(self, query: str) -> Dict:
        start_time = datetime.now()
        
        # Kiểm tra cache
        cached = self.cache.get(query)
        if cached:
            response_time = (datetime.now() - start_time).total_seconds()
            self.logger.log_response(query, cached['response'], 'cache', response_time)
            return {
                'response': cached['response'],
                'source': 'cache',
                'response_time': response_time
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
            response_time = (datetime.now() - start_time).total_seconds()
            
            self.cache.add(query, answer)
            self.logger.log_response(query, answer, 'api', response_time)
            
            return {
                'response': answer,
                'source': 'api',
                'response_time': response_time
            }

        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self.logger.log_response(query, str(e), 'error', response_time)
            raise Exception(f"Error generating response: {str(e)}")
