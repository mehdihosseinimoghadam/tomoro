import os
import pickle
import numpy as np
import tqdm
from nltk.tokenize import word_tokenize
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from langchain.docstore.document import Document
from src.core.database import create_database, save_embeddings, load_embeddings
from src.utils.helpers import embed_text, split_into_chunks, format_table
from src.config import DB_NAME
import nltk

nltk.download('punkt_tab')


def create_embeddings_and_bm25(data: List[Dict]) -> Tuple[List[np.ndarray], List[np.ndarray], List[Document], BM25Okapi]:
    create_database()

    pre_embeddings = []
    post_embeddings = []
    documents = []
    corpus = []
    embeddings_data = []

    print("Creating embeddings and BM25 index...")
    for i, item in enumerate(data):
        if i % 5 == 0:
            print(f"Processing item {i}/{len(data)}")
        pre_text = ' '.join(item['pre_text'])
        post_text = ' '.join(item['post_text'])
        table_text = format_table(item['table'])

        pre_doc = f"{pre_text}"
        post_doc = f"{post_text}"
        full_doc = f"{pre_doc}\n\n{post_doc}"

        # Split document into smaller chunks
        chunks = split_into_chunks(full_doc, chunk_size=500, overlap=100)

        for i, chunk in enumerate(chunks):
            # Add table to each chunk
            chunk_with_table = f"{chunk}\n\nTable:\n{table_text}"

            pre_embedding = embed_text(chunk_with_table)
            post_embedding = embed_text(chunk_with_table)

            doc_id = f"{item['id']}-{i}"
            pre_embeddings.append(pre_embedding)
            post_embeddings.append(post_embedding)
            documents.append(
                Document(page_content=chunk_with_table, metadata={"id": doc_id}))
            corpus.append(chunk_with_table)

            embeddings_data.append((doc_id, pickle.dumps(
                pre_embedding), pickle.dumps(post_embedding), chunk_with_table))

    # Save embeddings to database
    save_embeddings(embeddings_data)

    # Create BM25 index
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    return pre_embeddings, post_embeddings, documents, bm25


def load_or_create_embeddings_and_bm25(data: List[Dict]) -> Tuple[List[np.ndarray], List[np.ndarray], List[Document], BM25Okapi]:
    if os.path.exists(DB_NAME):
        print("Loading embeddings from database...")
        loaded_data = load_embeddings()
        pre_embeddings = [item[1] for item in loaded_data]
        post_embeddings = [item[2] for item in loaded_data]
        documents = [Document(page_content=item[3], metadata={
                              "id": item[0]}) for item in loaded_data]
        corpus = [item[3] for item in loaded_data]

        # Create BM25 index
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        return pre_embeddings, post_embeddings, documents, bm25
    else:
        return create_embeddings_and_bm25(data)
