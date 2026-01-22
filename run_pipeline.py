import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.summarizer import ExtractiveSummarizer


if __name__ == "__main__":
    text = """Artificial Intelligence is transforming industries by enabling machines to learn from data.
    Machine learning allows systems to improve automatically through experience.
    Deep learning uses neural networks with multiple layers.
    AI is widely used in healthcare, finance, and autonomous vehicles.
    Ethical concerns include bias, job displacement, and privacy.
    """

    summarizer = ExtractiveSummarizer(top_n=3)
    summary = summarizer.generate_summary(text)

    print("\n--- GENERATED SUMMARY ---\n")
    print(summary)
