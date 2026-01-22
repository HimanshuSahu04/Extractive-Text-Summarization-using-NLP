from src.abstractive_summarizer import AbstractiveSummarizer

if __name__ == "__main__":

    text = """
    Artificial Intelligence is transforming industries by enabling machines to learn from data.
    Machine learning allows systems to improve automatically through experience.
    Deep learning uses neural networks with multiple layers.
    AI is widely used in healthcare, finance, education, and autonomous vehicles.
    Ethical concerns include bias, job displacement, and privacy.
    Researchers are actively working on responsible AI.
    """

    summarizer = AbstractiveSummarizer()
    summary = summarizer.generate_summary(text)

    print("\n--- ABSTRACTIVE SUMMARY ---\n")
    print(summary)
