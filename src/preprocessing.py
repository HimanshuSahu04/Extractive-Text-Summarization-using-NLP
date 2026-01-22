import re
import nltk
from nltk.tokenize import sent_tokenize
from src.logger import logger
from src.exceptions import SummarizationException

nltk.download('punkt')

def clean_sentence(sentence: str) -> str:
    try:
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-zA-Z ]', '', sentence)
        return sentence
    except Exception as e:
        logger.error(f"Error cleaning sentence: {e}")
        raise SummarizationException("Sentence cleaning failed")

def tokenize_sentences(text: str):
    try:
        sentences = sent_tokenize(text)
        return sentences
    except Exception as e:
        logger.error(f"Sentence tokenization failed: {e}")
        raise SummarizationException("Tokenization failed")
