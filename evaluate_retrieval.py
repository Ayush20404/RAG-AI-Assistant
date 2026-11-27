
# evaluate_retrieval.py

# Test queries and manually identified relevant chunk IDs

import numpy as np

from pdf_loader import load_pdf, load_pdf_pages
from chunker import chunk_text, chunk_pages
from embedder import create_embeddings
from vector_store import create_vector_store

evaluation_dataset = [
    {
        "query": "Which technologies were used in the RAG AI Assistant project?",
        "relevant_ids": [1, 2]
    },
    {
        "query": "What is the candidate's educational qualification?",
        "relevant_ids": [0, 1]
    },
    {
        "query": "Which deep learning technologies were used in the Lung Cancer Detection project?",
        "relevant_ids": [4]
    },
    {
        "query": "Which skills are listed in the resume?",
        "relevant_ids": [1, 2, 3, 4]
    },
    {
        "query": "What technologies were used across the candidate's projects?",
        "relevant_ids": [1, 2, 3, 4]
    }
]


def calculate_precision(retrieved_ids, relevant_ids):
    """
    Precision = Relevant retrieved chunks / Total retrieved chunks
    """

    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)

    if not retrieved:
        return 0.0

    relevant_retrieved = retrieved.intersection(relevant)

    return len(relevant_retrieved) / len(retrieved)


def calculate_recall(retrieved_ids, relevant_ids):
    """
    Recall = Relevant retrieved chunks / Total relevant chunks
    """

    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    relevant_retrieved = retrieved.intersection(relevant)

    return len(relevant_retrieved) / len(relevant)



if __name__ == "__main__":

    print("\n========== RETRIEVAL EVALUATION ==========")

    # 1. Load PDF
    pdf_path = "uploaded.pdf"

    print("\nLoading PDF...")

    text = load_pdf(pdf_path)
    pages = load_pdf_pages(pdf_path)

    # 2. Create chunks (same methods as chatbot.py)
    print("Creating chunks...")

    chunks = chunk_text(text)
    page_chunks = chunk_pages(pages)

    print(f"Total chunks: {len(chunks)}")

    # Verify chunk alignment
    assert len(chunks) == len(page_chunks), (
        "Chunk count mismatch!"
    )

    for idx in range(len(chunks)):
        assert chunks[idx] == page_chunks[idx]["text"], (
            f"Chunk mismatch at index {idx}"
        )

    print("Chunk alignment verified.")

    # 3. Generate document embeddings
    print("\nGenerating document embeddings...")

    embeddings = create_embeddings(chunks)

    # 4. Build FAISS index
    print("\nBuilding FAISS index...")

    index = create_vector_store(embeddings)

    print("Vector store ready.")

    # 5. Generate all query embeddings in one batch
    queries = [
        item["query"] for item in evaluation_dataset
    ]

    print("\nGenerating query embeddings...")

    query_embeddings = create_embeddings(queries)

    # 6. Evaluate different top_k values
    
# 6. Evaluate different distance thresholds

threshold_values = [
    0.65, 0.70, 0.75, 0.80,
    0.85, 0.90, 0.95, 1.00
]

top_k = 5

print("\n========== THRESHOLD EVALUATION ==========")

# Search once per query, then apply different thresholds
all_search_results = []

for i, item in enumerate(evaluation_dataset):

    query_vector = np.array(
        [query_embeddings[i]]
    ).astype("float32")

    distances, indices = index.search(
        query_vector,
        top_k
    )

    all_search_results.append((distances[0], indices[0]))


for threshold in threshold_values:

    precision_scores = []
    recall_scores = []

    print(f"\n--- DISTANCE THRESHOLD = {threshold:.2f} ---")

    for i, item in enumerate(evaluation_dataset):

        distances, indices = all_search_results[i]

        retrieved_ids = [
            int(idx)
            for distance, idx in zip(distances, indices)
            if idx != -1 and distance <= threshold
        ]

        relevant_ids = item["relevant_ids"]

        precision = calculate_precision(
            retrieved_ids, relevant_ids
        )

        recall = calculate_recall(
            retrieved_ids, relevant_ids
        )

        precision_scores.append(precision)
        recall_scores.append(recall)

        print(f"\nQuery: {item['query']}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")

    avg_precision = (
        sum(precision_scores) / len(precision_scores)
    )

    avg_recall = (
        sum(recall_scores) / len(recall_scores)
    )

    print("\nAverage Results:")
    print(f"Precision: {avg_precision * 100:.2f}%")
    print(f"Recall: {avg_recall * 100:.2f}%")

print("\n========== THRESHOLD EVALUATION COMPLETE ==========")

