from config import client
import numpy as np


def retrieve_chunks(query, index, chunks, page_chunks, top_k=5, distance_threshold=1.0):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=query
    )

    query_embedding = response.embeddings[0].values

    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(
        query_vector,
        top_k
    )

    retrieved_chunks = []

    for distance, idx in zip(distances[0], indices[0]):

        if idx != -1 and distance <= distance_threshold:
            retrieved_chunks.append({
                "text": page_chunks[idx]["text"],
                "page": page_chunks[idx]["page"]
            })

    return retrieved_chunks