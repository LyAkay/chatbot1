def preprocess_text(text: str) -> str:
    """
    Tiền xử lý văn bản, loại bỏ ký tự không cần thiết.
    """
    return text.strip().lower()
