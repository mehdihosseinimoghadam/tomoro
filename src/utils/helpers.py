import numpy as np
from typing import List, Dict
import json
from openai import OpenAI
from src.config import OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_text(text: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


def load_data(file_path: str) -> List[Dict]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def generate_text(prompt: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def format_table(table: List[List[str]]) -> str:
    formatted_table = []
    for row in table:
        formatted_row = " | ".join(row)
        formatted_table.append(formatted_row)
    return "\n".join(formatted_table)


def query_expansion(query: str) -> List[str]:
    expansion_prompt = f"""
    Given the following query, generate 2 alternative phrasings or related queries that might help in retrieving relevant information:

    Original query: {query}

    Alternative queries:
    1.
    2.
    """
    response = generate_text(expansion_prompt)
    expanded_queries = response.strip().split('\n')[1:]
    expanded_queries = [q.split('.', 1)[1].strip() for q in expanded_queries]
    return [query] + expanded_queries
