from typing import List, Dict
import numpy as np
import tqdm
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
from ragas import evaluate
from datasets import Dataset
from src.core.rag import rag_system
from ragas.metrics import (
    context_precision,
    faithfulness,
    answer_relevancy,
    context_recall,
    context_utilization,
    answer_correctness
)


def benchmark_rag_system(benchmarking_data: List[Dict], pre_embeddings: List[np.ndarray], post_embeddings: List[np.ndarray], documents: List[Document], bm25: BM25Okapi) -> Dict[str, float]:
    results = []

    for i, item in enumerate(benchmarking_data):
        print(f"Processing item {i}/{len(benchmarking_data)}")
        query = item['question']
        ground_truth = item['answer']
        reference_context = item.get('context', '')

        # Ensure reference_context is a string
        if isinstance(reference_context, list):
            reference_context = ' '.join(reference_context)
        elif not isinstance(reference_context, str):
            reference_context = str(reference_context)

        try:
            rag_result = rag_system(
                query, pre_embeddings, post_embeddings, documents, bm25)
            rag_result['ground_truths'] = [ground_truth]
            rag_result['reference'] = reference_context
            results.append(rag_result)

        except Exception as e:
            print(f"Error processing query: {query}")
            print(f"Error message: {str(e)}")
            continue

    # Convert results to a Dataset
    dataset = Dataset.from_list(results)

    # Evaluate using RAGAS
    evaluation_result = evaluate(
        dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
            context_utilization,
            answer_correctness
        ]
    )

    return evaluation_result
