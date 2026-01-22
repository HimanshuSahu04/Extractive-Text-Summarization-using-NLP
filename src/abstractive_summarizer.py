import torch
from transformers import pipeline
from src.logger import logger
from src.exceptions import SummarizationException


class AbstractiveSummarizer:
    """
    Custom abstractive summarizer with extractive-like control.
    You control summary size using `sentences`, not raw tokens.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device=None
    ):
        try:
            if device is None:
                device = 0 if torch.cuda.is_available() else -1

            logger.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")

            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=device
            )

        except Exception as e:
            logger.exception("Failed to load abstractive model")
            raise SummarizationException(str(e))

    def generate_summary(self, text: str, sentences: int = 3) -> str:
        """
        sentences: number of sentence-equivalents you want in the summary
        """

        try:
            # 🔑 sentence → token mapping
            max_length = sentences * 30
            min_length = max(15, sentences * 20)

            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )

            return summary[0]["summary_text"]

        except Exception as e:
            logger.exception("Abstractive summarization failed")
            raise SummarizationException(str(e))
