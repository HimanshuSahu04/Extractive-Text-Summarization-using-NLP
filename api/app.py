from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.logger import logger
from src.exceptions import SummarizationException

app = FastAPI(
    title="Extractive Text Summarization API",
    version="1.0"
)

class SummarizeRequest(BaseModel):
    text: str
    sentences: int = 3
    method: str = "extractive"  # extractive | abstractive


class SummarizeResponse(BaseModel):
    summary: str

@app.post("/summarize")
def summarize_text(request: SummarizeRequest):

    if request.method == "extractive":
        summarizer = ExtractiveSummarizer(top_n=request.sentences)
        summary = summarizer.generate_summary(request.text)

    elif request.method == "abstractive":
        summarizer = AbstractiveSummarizer()
        summary = summarizer.generate_summary(request.text)

    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid method. Use 'extractive' or 'abstractive'"
        )

    return {"summary": summary}
