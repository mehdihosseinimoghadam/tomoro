from typing import List, Dict
import numpy as np
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
from src.utils.helpers import cosine_similarity, generate_text, query_expansion, embed_text
from langchain.prompts import PromptTemplate
from nltk.tokenize import word_tokenize


def hybrid_retrieve_documents(query: str, pre_embeddings: List[np.ndarray], post_embeddings: List[np.ndarray], documents: List[Document], bm25: BM25Okapi, k: int = 10) -> List[Document]:
    query_embedding = embed_text(query)

    # Embedding-based similarity
    similarities = []
    for pre_emb, post_emb in zip(pre_embeddings, post_embeddings):
        pre_sim = cosine_similarity(query_embedding, pre_emb)
        post_sim = cosine_similarity(query_embedding, post_emb)
        similarities.append(max(pre_sim, post_sim))

    # BM25 scores
    tokenized_query = word_tokenize(query.lower())
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize scores
    max_similarity = max(similarities)
    max_bm25 = max(bm25_scores)
    normalized_similarities = [sim / max_similarity for sim in similarities]
    normalized_bm25_scores = [score / max_bm25 for score in bm25_scores]

    # Combine scores (you can adjust the weights)
    combined_scores = [0.7 * sim + 0.3 * bm25 for sim,
                       bm25 in zip(normalized_similarities, normalized_bm25_scores)]

    top_k_indices = np.argsort(combined_scores)[-k:][::-1]
    return [documents[i] for i in top_k_indices]


def rerank_documents(query: str, documents: List[Document], k: int = 5) -> List[Document]:
    rerank_prompt = """
    Given the following query and document, rate the relevance of the document to the query on a scale of 0 to 10, where 0 is completely irrelevant and 10 is highly relevant. Provide a brief explanation for your rating.

    Query: {query}

    Document:
    {document}

    Relevance score (0-10) and explanation:
    """

    scored_docs = []
    for doc in documents:
        prompt = rerank_prompt.format(query=query, document=doc.page_content)
        response = generate_text(prompt)
        try:
            score_text, explanation = response.split('\n', 1)
            score = float(score_text.strip().split(':')[1])
        except ValueError:
            score = 0
            explanation = "Failed to parse score"
        scored_docs.append((score, explanation, doc))

    # Sort documents by score and return top k
    reranked_docs = [doc for score, explanation, doc in sorted(
        scored_docs, key=lambda x: x[0], reverse=True)]
    return reranked_docs[:k]


def generate_answer(query: str, context: str) -> str:
    prompt_template = """
    You are an AI assistant tasked with answering questions based on the given context. Follow these steps:

    1. Carefully read the context and the question.
    2. Identify the key information in the context that is relevant to the question.
    3. Formulate a clear and concise answer based on the relevant information.
    4. If the context doesn't contain enough information to answer the question confidently, say "I don't have enough information to answer this question accurately."
    5. Provide your confidence level in the answer (low, medium, or high).
    6. Provide the explanation for the answer in Explanatgion: part and the final result in Answer: part (do not add any other text other than the answer in Answer: part).

    Context:
    {context}

    Question: {query}

    Explanation:
    
    
    Answer:
    """

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "query"])
    full_prompt = prompt.format(context=context, query=query)

    return generate_text(full_prompt)


def rag_system(query: str, pre_embeddings: List[np.ndarray], post_embeddings: List[np.ndarray], documents: List[Document], bm25: BM25Okapi) -> Dict[str, str]:
    expanded_queries = query_expansion(query)
    all_retrieved_docs = []

    for expanded_query in expanded_queries:
        retrieved_docs = hybrid_retrieve_documents(
            expanded_query, pre_embeddings, post_embeddings, documents, bm25, k=10)
        all_retrieved_docs.extend(retrieved_docs)

    unique_docs = list(
        {doc.metadata['id']: doc for doc in all_retrieved_docs}.values())

    reranked_docs = rerank_documents(query, unique_docs, k=5)
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    answer = generate_answer(query, context)

    return {
        "question": query,
        "answer": answer,
        "contexts": [doc.page_content for doc in reranked_docs]
    }
