from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.summarizer import ExtractiveSummarizer
from src.logger import logger
from src.exceptions import SummarizationException

app = FastAPI(
    title="Extractive Text Summarization API",
    version="1.0"
)

class SummarizeRequest(BaseModel):
    text: str
    sentences: int = 3

class SummarizeResponse(BaseModel):
    summary: str

@app.post("/summarize", response_model=SummarizeResponse)
def summarize_text(request: SummarizeRequest):
    try:
        summarizer = ExtractiveSummarizer(top_n=request.sentences)
        summary = summarizer.generate_summary(request.text)
        return {"summary": summary}

    except SummarizationException as e:
        logger.error(str(e))
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail="Internal Server Error")
