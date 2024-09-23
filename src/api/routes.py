from fastapi import APIRouter, Depends
from src.api.models import QueryInput, RAGResponse, EvaluationResult
from src.core.rag import rag_system
from src.core.evaluation import benchmark_rag_system
from src.core.embeddings import load_or_create_embeddings_and_bm25
from src.config import DATA_PATH
from src.utils.helpers import load_data
from src.data.benchmarking import benchmarking_data

data = load_data(DATA_PATH)
router = APIRouter()


@router.post("/rag", response_model=RAGResponse)
async def perform_rag(query_input: QueryInput):
    pre_embeddings, post_embeddings, documents, bm25 = load_or_create_embeddings_and_bm25(
        data)
    result = rag_system(query_input.query, pre_embeddings,
                        post_embeddings, documents, bm25)
    return RAGResponse(**result)


@router.post("/evaluate", response_model=EvaluationResult)
async def evaluate_rag():
    pre_embeddings, post_embeddings, documents, bm25 = load_or_create_embeddings_and_bm25(
        data)
    result = benchmark_rag_system(
        benchmarking_data, pre_embeddings, post_embeddings, documents, bm25)
    return EvaluationResult(metrics=result)
