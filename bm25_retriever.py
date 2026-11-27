
from rank_bm25 import BM25Okapi
import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()


def tokenize_text(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [stemmer.stem(token) for token in tokens]



def create_bm25_index(page_chunks):
    tokenized_chunks = [
        tokenize_text(chunk["text"])
        for chunk in page_chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25



def retrieve_bm25(query, bm25_index, page_chunks, top_k=5):
    tokenized_query = tokenize_text(query)

    scores = bm25_index.get_scores(tokenized_query)

    top_indices = scores.argsort()[::-1][:top_k]

    retrieved_chunks = []

    for idx in top_indices:
        if scores[idx] > 0:
            retrieved_chunks.append({
                "chunk_id": int(idx),
                "text": page_chunks[idx]["text"],
                "page": page_chunks[idx]["page"],
                "bm25_score": float(scores[idx])
            })

    return retrieved_chunks
