from rouge_score import rouge_scorer
from src.logger import logger
from src.exceptions import SummarizationException

def evaluate_summary(reference: str, generated: str):
    try:
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )

        scores = scorer.score(reference, generated)

        results = {
            "ROUGE-1": scores["rouge1"].fmeasure,
            "ROUGE-2": scores["rouge2"].fmeasure,
            "ROUGE-L": scores["rougeL"].fmeasure
        }

        logger.info(f"Evaluation scores: {results}")
        return results

    except Exception as e:
        logger.error(f"ROUGE evaluation failed: {e}")
        raise SummarizationException("Evaluation failed")
