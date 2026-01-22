class SummarizationException(Exception):
    """Custom exception class for summarization errors"""
    def __init__(self, message: str):
        super().__init__(message)
