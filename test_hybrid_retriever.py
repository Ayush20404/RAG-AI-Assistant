
from pdf_loader import load_pdf_pages
from chunker import chunk_pages
from embedder import create_embeddings
from vector_store import create_vector_store
from bm25_retriever import create_bm25_index
from hybrid_retriever import retrieve_hybrid


# 1. Load PDF
pdf_path = "uploaded.pdf"

print("\nLoading PDF...")
pages = load_pdf_pages(pdf_path)


# 2. Create page-aware chunks
print("\nCreating chunks...")
page_chunks = chunk_pages(pages)

chunks = [chunk["text"] for chunk in page_chunks]

print(f"Total chunks: {len(page_chunks)}")


# 3. Build dense retrieval index
print("\nGenerating embeddings...")
embeddings = create_embeddings(chunks)

print("\nBuilding FAISS index...")
index = create_vector_store(embeddings)


# 4. Build BM25 index
print("\nBuilding BM25 index...")
bm25_index = create_bm25_index(page_chunks)


# 5. Test hybrid retrieval
query = "Which technologies were used in the RAG AI Assistant project?"

print(f"\nQuery: {query}")

results = retrieve_hybrid(
    query=query,
    index=index,
    bm25_index=bm25_index,
    chunks=chunks,
    page_chunks=page_chunks,
    top_k=5,
    dense_top_k=5,
    bm25_top_k=5,
    distance_threshold=1.0
)


# 6. Display results
print("\n========== HYBRID RETRIEVAL RESULTS ==========")

for rank, chunk in enumerate(results, start=1):
    print(f"\nRank: {rank}")
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page']}")
    print(f"RRF Score: {chunk['rrf_score']:.6f}")
    print(f"Text: {chunk['text'][:250]}")
    print("-" * 60)
