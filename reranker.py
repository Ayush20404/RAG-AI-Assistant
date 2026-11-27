
from sentence_transformers import CrossEncoder

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

reranker = CrossEncoder(RERANKER_MODEL)


def rerank_chunks(query, retrieved_chunks, top_n=3):

    if not retrieved_chunks:
        return []

    pairs = [
        [query, chunk["text"]]
        for chunk in retrieved_chunks
    ]

    scores = reranker.predict(pairs)

    ranked_chunks = []

    for chunk, score in zip(retrieved_chunks, scores):
        ranked_chunks.append({
            **chunk,
            "rerank_score": float(score)
        })

    ranked_chunks.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return ranked_chunks[:top_n]
