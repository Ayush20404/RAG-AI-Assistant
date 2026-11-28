
import numpy as np

from pdf_loader import load_pdf, load_pdf_pages
from chunker import chunk_text, chunk_pages
from embedder import create_embeddings
from vector_store import create_vector_store

from bm25_retriever import create_bm25_index, retrieve_bm25
from retriever import retrieve_chunks
from hybrid_retriever import retrieve_hybrid



# -----------------------------------------
# 1. Evaluation queries and ground truth
# -----------------------------------------

evaluation_queries = [
    {
        "query": "Which technologies were used in the RAG AI Assistant project?",
        "relevant_ids": [2]
    },
    {
        "query": "What are the educational qualifications?",
        "relevant_ids": [1]
    },
    {
        "query": "Which technologies were used in the Lung Cancer Detection project?",
        "relevant_ids": [4]
    },
    {
        "query": "Which technologies were used across the projects?",
        "relevant_ids": [2, 3, 4]
    }
]

top_k = 3
distance_threshold = 1.0



# -----------------------------------------
# 2. Load PDF and create chunks
# -----------------------------------------

print("\nLoading PDF...")

pdf_path = "uploaded.pdf"

text = load_pdf(pdf_path)
pages = load_pdf_pages(pdf_path)

print("\nCreating chunks...")

chunks = chunk_text(text)
page_chunks = chunk_pages(pages)

for chunk in page_chunks:
    chunk["source"] = pdf_path
    chunk["document_type"] = "pdf"

assert len(chunks) == len(page_chunks), "Chunk count mismatch!"

for idx, chunk in enumerate(chunks):
    assert chunk == page_chunks[idx]["text"], (
        f"Chunk mismatch at index {idx}"
    )

print(f"Total chunks: {len(chunks)}")

print("\n========== CHUNK MAPPING ==========")

for idx, chunk in enumerate(page_chunks):
    print(f"\nChunk ID: {idx}")
    print(f"Page: {chunk['page']}")
    print(f"Text: {chunk['text'][:200]}")
    print("-" * 50)


# -----------------------------------------
# 3. Build Dense Retrieval index
# -----------------------------------------

print("\nGenerating document embeddings...")

embeddings = create_embeddings(chunks)

print("\nBuilding FAISS index...")

dense_index = create_vector_store(embeddings)


# Metadata filtering test

print("\nTesting Metadata Filtering...")

filtered_results = retrieve_chunks(
    query="Which technologies were used in the RAG AI Assistant project?",
    index=dense_index,
    chunks=chunks,
    page_chunks=page_chunks,
    metadata_filter={
        "source": pdf_path,
        "document_type": "pdf"
    }
)

print("\nFiltered Results:")

for item in filtered_results:
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Source: {item['source']}")
    print(f"Document Type: {item['document_type']}")
    print(f"Page: {item['page']}")

assert all(
    item["source"] == pdf_path
    and item["document_type"] == "pdf"
    for item in filtered_results
)

print("\nMetadata filtering test passed!")


# -----------------------------------------
# 4. Build BM25 index
# -----------------------------------------

print("\nBuilding BM25 index...")

bm25_index = create_bm25_index(page_chunks)



# -----------------------------------------
# 5. Evaluation metrics
# -----------------------------------------

def calculate_precision(retrieved_ids, relevant_ids):
    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)

    if not retrieved:
        return 0.0

    return len(retrieved.intersection(relevant)) / len(retrieved)


def calculate_recall(retrieved_ids, relevant_ids):
    retrieved = set(retrieved_ids)
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    return len(retrieved.intersection(relevant)) / len(relevant)


def calculate_reciprocal_rank(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1 / rank

    return 0.0


# -----------------------------------------
# 6. Run evaluation across all queries
# -----------------------------------------

all_metrics = {
    "DENSE (FAISS)": [],
    "BM25": [],
    "HYBRID (RRF)": []
}

for test_case in evaluation_queries:

    query = test_case["query"]
    relevant_ids = test_case["relevant_ids"]

    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print(f"Relevant Chunk IDs: {relevant_ids}")
    print("=" * 60)

    # Dense retrieval
    dense_results = retrieve_chunks(
        query,
        dense_index,
        chunks,
        page_chunks,
        top_k=top_k,
        distance_threshold=distance_threshold
    )

    # BM25 retrieval
    bm25_results = retrieve_bm25(
        query,
        bm25_index,
        page_chunks,
        top_k=top_k
    )

    # Hybrid retrieval
    hybrid_results = retrieve_hybrid(
        query,
        dense_index,
        bm25_index,
        chunks,
        page_chunks,
        top_k=top_k,
        dense_top_k=top_k,
        bm25_top_k=top_k,
        distance_threshold=distance_threshold
    )

    retrieval_methods = {
        "DENSE (FAISS)": dense_results,
        "BM25": bm25_results,
        "HYBRID (RRF)": hybrid_results
    }

    for method_name, results in retrieval_methods.items():

        retrieved_ids = [
            item["chunk_id"] for item in results
        ]

        precision = calculate_precision(
            retrieved_ids, relevant_ids
        )

        recall = calculate_recall(
            retrieved_ids, relevant_ids
        )

        reciprocal_rank = calculate_reciprocal_rank(
            retrieved_ids, relevant_ids
        )

        all_metrics[method_name].append({
            "precision": precision,
            "recall": recall,
            "rr": reciprocal_rank
        })

        print(f"\nMethod: {method_name}")
        print(f"Retrieved IDs: {retrieved_ids}")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"Reciprocal Rank: {reciprocal_rank:.3f}")


# -----------------------------------------
# 7. Calculate average evaluation scores
# -----------------------------------------

print("\n\n" + "=" * 65)
print("FINAL RETRIEVAL EVALUATION")
print("=" * 65)

for method_name, metrics in all_metrics.items():

    avg_precision = np.mean([
        item["precision"] for item in metrics
    ])

    avg_recall = np.mean([
        item["recall"] for item in metrics
    ])

    mrr = np.mean([
        item["rr"] for item in metrics
    ])

    print(f"\nMethod: {method_name}")
    print(f"Average Precision: {avg_precision * 100:.2f}%")
    print(f"Average Recall: {avg_recall * 100:.2f}%")
    print(f"MRR: {mrr:.3f}")

print("\n========== EVALUATION COMPLETE ==========")
