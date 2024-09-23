from pydantic import BaseModel
from typing import List, Dict


class QueryInput(BaseModel):
    query: str


class RAGResponse(BaseModel):
    question: str
    answer: str
    contexts: List[str]


class EvaluationResult(BaseModel):
    metrics: Dict[str, float]
