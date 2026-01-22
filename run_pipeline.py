from src.summarizer import ExtractiveSummarizer
from src.evaluation import evaluate_summary

if __name__ == "__main__":

    text = """
    Artificial Intelligence is transforming industries by enabling machines to learn from data.
    Machine learning allows systems to improve automatically through experience.
    Deep learning uses neural networks with multiple layers.
    AI is widely used in healthcare, finance, and autonomous vehicles.
    Ethical concerns include bias, job displacement, and privacy.
    """

    reference_summary = """
    Artificial Intelligence is transforming industries.
    AI is widely used in healthcare, finance, and autonomous vehicles.
    Ethical concerns include bias and privacy.
    """

    summarizer = ExtractiveSummarizer(top_n=3)
    generated_summary = summarizer.generate_summary(text)

    scores = evaluate_summary(reference_summary, generated_summary)

    print("\n--- GENERATED SUMMARY ---\n")
    print(generated_summary)

    print("\n--- ROUGE SCORES ---\n")
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")
