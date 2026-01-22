import numpy as np
from src.preprocessing import clean_sentence, tokenize_sentences
from src.vectorizer import create_tfidf_matrix
from src.logger import logger
from src.exceptions import SummarizationException

class ExtractiveSummarizer:

    def __init__(self, top_n: int = 3):
        self.top_n = top_n

    def generate_summary(self, text: str) -> str:
        try:
            logger.info("Starting text summarization")

            sentences = tokenize_sentences(text)
            cleaned_sentences = [clean_sentence(s) for s in sentences]

            tfidf_matrix = create_tfidf_matrix(cleaned_sentences)
            scores = np.sum(tfidf_matrix.toarray(), axis=1)

            ranked_indices = np.argsort(scores)[::-1][:self.top_n]
            ranked_indices = sorted(ranked_indices)

            summary = " ".join([sentences[i] for i in ranked_indices])

            logger.info("Summarization completed successfully")
            return summary

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise SummarizationException("Text summarization failed")
