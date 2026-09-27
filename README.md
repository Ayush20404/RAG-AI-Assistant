# 🤖 RAG AI Assistant

A Generative AI assistant built with Python, Gemini API, Streamlit, and FAISS that can handle general conversations as well as question answering over uploaded PDF documents.

The project implements a Retrieval-Augmented Generation (RAG) pipeline with semantic search, conversational memory, query rewriting, agent-based routing, and source citations.

---

## ✨ Features

- 💬 General conversational queries
- 📄 Upload and process PDF documents
- 🔍 Semantic search using vector embeddings
- 🧠 Retrieval-Augmented Generation (RAG)
- 🔄 Query rewriting for conversational follow-up questions
- 🤖 Agent-based routing between general and document-related queries
- 💾 Short-term conversation memory
- 🧠 Long-term user memory
- 📚 Page-based source citations in document answers
- 🗑️ Clear chat and conversation history
- 🎨 Streamlit-based interactive UI
- ⚡ FAISS vector store for efficient similarity search

---

## 🛠️ Tech Stack

- Python
- Google Gemini API
- Streamlit
- FAISS
- NumPy
- PyPDF
- python-dotenv

---

## 🚂 RAG Pipeline

The document-based question answering flow follows:

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Text Chunking
    ↓
Gemini Embeddings
    ↓
FAISS Vector Store
    ↓
Query Rewriting
    ↓
Semantic Retrieval
    ↓
Relevant Context
    ↓
Gemini LLM
    ↓
Answer + Source Citations
```

---

## 📂 Project Structure

```text
RAG-AI-Assistant/
│
├── app.py                  # Streamlit user interface
├── chatbot.py              # Main chatbot orchestration
├── agent.py                # Query/action routing
├── config.py               # Gemini API configuration
│
├── pdf_loader.py           # PDF text extraction
├── chunker.py              # Text chunking
├── embedder.py             # Generate embeddings
├── vector_store.py         # FAISS vector store
├── retriever.py            # Semantic retrieval
├── query_rewritter.py       # Conversational query rewriting
│
├── memory.py               # Short-term conversation memory
├── long_term_memory.py     # Long-term memory handling
├── models.py               # Model configuration
├── prompts.py              # Prompt templates
├── utils.py                # Utility functions
│
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd RAG-AI-Assistant
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
```

> Never commit your `.env` file or API key to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📖 How It Works

The assistant first determines whether a user's query is a general question or requires information from an uploaded document.

For document-related questions, the system:

1. Processes the uploaded PDF.
2. Splits the extracted text into overlapping chunks.
3. Generates embeddings for the chunks.
4. Stores the embeddings in FAISS.
5. Rewrites conversational queries when necessary.
6. Retrieves the most relevant chunks.
7. Sends the retrieved context to Gemini.
8. Generates an answer with relevant PDF page citations.

For general questions, the assistant can respond without requiring a PDF.

---

## 🔐 Security

API keys and runtime-generated files are kept outside the GitHub repository.

The following files are ignored using `.gitignore`:

```text
.env
memory.json
long_term_memory.json
uploaded.pdf
```

---

## 🎯 Project Goal

This project was built as a hands-on exploration of Generative AI and Retrieval-Augmented Generation, focusing on understanding how different components of an AI assistant work together in practice.