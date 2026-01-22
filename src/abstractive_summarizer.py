from transformers import pipeline
from src.logger import logger
from src.exceptions import SummarizationException


class AbstractiveSummarizer:

    def __init__(
        self,
        model_name: str = "sshleifer/distilbart-cnn-12-6",
        max_length: int = 130,
        min_length: int = 30
    ):
        try:
            logger.info("Loading abstractive summarization model")
            self.summarizer = pipeline(
                "summarization",
                model=model_name
            )
            self.max_length = max_length
            self.min_length = min_length

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise SummarizationException("Failed to load abstractive model")

    def generate_summary(self, text: str) -> str:
        try:
            logger.info("Generating abstractive summary")

            summary = self.summarizer(
                text,
                max_length=self.max_length,
                min_length=self.min_length,
                do_sample=False
            )

            return summary[0]["summary_text"]

        except Exception as e:
            logger.error(f"Abstractive summarization failed: {e}")
            raise SummarizationException("Abstractive summarization failed")
