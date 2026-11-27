import numpy as np
from embedder import create_embeddings

from models import EMBEDDING_MODEL


def retrieve_chunks(query, index, chunks, page_chunks, top_k=5, distance_threshold=1.0):

    query_embedding = create_embeddings([query])[0]

    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(
        query_vector,
        top_k
    )

    print("\n--- RETRIEVAL DIAGNOSTICS ---")

    for rank, (distance, idx) in enumerate(
        zip(distances[0], indices[0]),
        start=1
    ):
        if idx != -1:
            print(f"Rank: {rank}")
            print(f"FAISS Index: {idx}")
            print(f"Squared L2 Distance: {distance:.4f}")
            print(f"Chunk: {page_chunks[idx]['text'][:150]}")
            print("-" * 50)

    retrieved_chunks = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx != -1 and distance <= distance_threshold:

            retrieved_chunks.append({
                "chunk_id": int(idx),
                "text": page_chunks[idx]["text"],
                "page": page_chunks[idx]["page"],
                "distance": float(distance)
            })

    return retrieved_chunks