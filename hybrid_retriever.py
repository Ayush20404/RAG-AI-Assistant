from retriever import retrieve_chunks
from bm25_retriever import retrieve_bm25


def reciprocal_rank_fusion(
    dense_results,
    bm25_results,
    k=60,
    top_k=5
):
    scores = {}
    chunk_data = {}

    # Process dense retrieval results
    for rank, chunk in enumerate(dense_results, start=1):
        chunk_id = chunk["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        chunk_data[chunk_id] = chunk

    # Process BM25 retrieval results
    for rank, chunk in enumerate(bm25_results, start=1):
        chunk_id = chunk["chunk_id"]

        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)
        chunk_data[chunk_id] = chunk

    # Sort chunks by their combined RRF score
    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    hybrid_results = []

    for chunk_id in ranked_ids[:top_k]:
        chunk = chunk_data[chunk_id].copy()
        chunk["rrf_score"] = scores[chunk_id]
        hybrid_results.append(chunk)

    return hybrid_results


def retrieve_hybrid(
    query,
    index,
    bm25_index,
    chunks,
    page_chunks,
    top_k=5,
    dense_top_k=5,
    bm25_top_k=5,
    distance_threshold=1.0
):
    dense_results = retrieve_chunks(
        query,
        index,
        chunks,
        page_chunks,
        top_k=dense_top_k,
        distance_threshold=distance_threshold
    )

    bm25_results = retrieve_bm25(
        query,
        bm25_index,
        page_chunks,
        top_k=bm25_top_k
    )

    hybrid_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        top_k=top_k
    )

    return hybrid_results