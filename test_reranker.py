
from reranker import rerank_chunks

query = "Which deep learning technologies were used in the Lung Cancer Detection project?"

retrieved_chunks = [
    {
        "text": "AI Shopping Assistant: Python, Streamlit, Pandas, Scikit-learn.",
        "page": 1
    },
    {
        "text": "Lung Cancer Detection: Python, TensorFlow, Keras, VGG16, Deep Learning, Image Processing.",
        "page": 1
    },
    {
        "text": "RAG AI Assistant: Gemini API, Streamlit, FAISS, Embeddings.",
        "page": 1
    },
    {
        "text": "Lung Cancer Detection achieved 94.55% test accuracy using data augmentation.",
        "page": 1
    }
]

print("\n========== BEFORE RERANKING ==========")

for i, chunk in enumerate(retrieved_chunks, start=1):
    print(f"\nRank {i}:")
    print(chunk["text"])

ranked_chunks = rerank_chunks(
    query,
    retrieved_chunks,
    top_n=3
)

print("\n========== AFTER RERANKING ==========")

for i, chunk in enumerate(ranked_chunks, start=1):
    print(f"\nRank {i}:")
    print(f"Score: {chunk['rerank_score']:.4f}")
    print(chunk["text"])
