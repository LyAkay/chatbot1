# Dự phòng cho các model cần thiết trong tương lai
class ChatResponse:
    def __init__(self, question: str, answer: str, context: str, timestamp: str):
        self.question = question
        self.answer = answer
        self.context = context
        self.timestamp = timestamp
