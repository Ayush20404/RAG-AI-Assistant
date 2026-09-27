
from memory import save_memory, load_memory
from long_term_memory import extract_memory, save_long_term_memory, load_long_term_memory

from config import client
from prompts import RAG_PROMPT

from pdf_loader import load_pdf, load_pdf_pages
from chunker import chunk_text, chunk_pages
from embedder import create_embeddings
from vector_store import create_vector_store
from retriever import retrieve_chunks
from models import CHAT_MODEL

from query_rewritter import rewrite_query

from agent import decide_action





class RAGChatbot:

    def __init__(self, pdf_path=None):

        self.history = load_memory()

        if pdf_path is not None:

            print("\nLoading PDF...")

            self.text = load_pdf(pdf_path)

            print("✓ PDF Loaded")

            self.pages = load_pdf_pages(pdf_path)

            print("\nCreating Chunks...")

            self.chunks = chunk_text(self.text)

            print(f"✓ {len(self.chunks)} Chunks Created")

            self.page_chunks = chunk_pages(self.pages)

            print("\nGenerating Embeddings...")

            self.embeddings = create_embeddings(self.chunks)

            print("✓ Embeddings Generated")

            print("\nBuilding Vector Store...")

            self.index = create_vector_store(self.embeddings)

            print("✓ Vector Store Ready")

    def chat(self, query):

        action = decide_action(query)

        print(f"Agent decision: {action}")

        self.history.append({
            "role": "user",
            "content": query
        })

        recent_history = self.history[-6:]

        long_term_memory = load_long_term_memory()

        memory_context = ""

        for key, value in long_term_memory.items():
            memory_context += f"{key}: {value}\n"

        if action == "GENERAL_QUERY":

            conversation = ""

            conversation += f"User information:\n{memory_context}\n"


            for message in recent_history:
                conversation += f"{message['role']}: {message['content']}\n"

            response = client.models.generate_content(
                model=CHAT_MODEL,
                contents=conversation
            )

            answer = response.text

           
            self.history.append({
                "role": "assistant",
                "content": answer
            })

            save_memory(self.history)

            memory = extract_memory(query)

            if memory:
                save_long_term_memory(memory)

            return answer
        

        if not hasattr(self, "index") or not hasattr(self, "chunks"):
            return "Please upload a PDF to ask questions about a document."
        

        rewritten_query = rewrite_query(
            query,
            self.history
        )

        print(f"\nOriginal query: {query}")
        print(f"Rewritten query: {rewritten_query}")

        retrieved_chunks = retrieve_chunks(
            query=rewritten_query,
            index=self.index,
            chunks=self.chunks,
            page_chunks=self.page_chunks
        )

        

        if not retrieved_chunks:
            return "I couldn't find this information in the uploaded PDF."

        

        context = "\n\n".join(
            f"Page {chunk['page']}:\n{chunk['text']}"
            for chunk in retrieved_chunks
        )

        conversation = ""


        for message in recent_history:
            conversation += f"{message['role']}: {message['content']}\n"


        final_prompt = RAG_PROMPT.format(
            context=context,
            question=rewritten_query,
            conversation=conversation,
            memory_context=memory_context

        )

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=final_prompt
        )

        answer = response.text


        self.history.append({
            "role": "assistant",
            "content": answer
        })

        save_memory(self.history)

        return answer