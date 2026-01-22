from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logger
from src.exceptions import SummarizationException

def create_tfidf_matrix(sentences: list):
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(sentences)
        return tfidf_matrix
    except Exception as e:
        logger.error(f"TF-IDF vectorization failed: {e}")
        raise SummarizationException("TF-IDF creation failed")
