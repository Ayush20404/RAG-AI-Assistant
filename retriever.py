
import numpy as np
from embedder import create_embeddings


def retrieve_chunks(
    query,
    index,
    chunks,
    page_chunks,
    top_k=5,
    distance_threshold=1.0,
    metadata_filter=None
):

    query_embedding = create_embeddings([query])[0]

    query_vector = np.array(
        [query_embedding]
    ).astype("float32")

    distances, indices = index.search(
        query_vector,
        index.ntotal
    )

    retrieved_chunks = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx == -1:
            continue

        if distance > distance_threshold:
            continue

        chunk = page_chunks[idx]

        # Apply metadata filtering
        if metadata_filter:
            matches = all(
                chunk.get(key) == value
                for key, value in metadata_filter.items()
            )

            if not matches:
                continue

        retrieved_chunks.append({
            "chunk_id": int(idx),
            "text": chunk["text"],
            "page": chunk["page"],
            "distance": float(distance),
            "source": chunk.get("source"),
            "document_type": chunk.get("document_type")
        })

        if len(retrieved_chunks) >= top_k:
            break

    return retrieved_chunks
