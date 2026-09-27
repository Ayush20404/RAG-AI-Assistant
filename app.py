
import streamlit as st
from chatbot import RAGChatbot
from memory import save_memory


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG-Based AI Assistant",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    /* ========================================================
       GLOBAL APP
       ======================================================== */

    .stApp {
        background-color: #0e1117;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    [data-testid="stSidebar"] {
        background-color: #171a21;
        border-right: 1px solid #262b35;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f1f3f5;
        font-weight: 600;
    }


    /* ========================================================
       MAIN HEADER
       ======================================================== */

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
        color: #f5f7fa;
    }

    .subtitle {
        color: #8f98a8;
        font-size: 1rem;
        margin-bottom: 2.5rem;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    [data-testid="stFileUploader"] {
        background-color: #101319;
        border: 1px solid #292f3a;
        border-radius: 12px;
        padding: 0.5rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #4b5565;
    }


    /* ========================================================
       DOCUMENT INFORMATION CARD
       ======================================================== */

    .document-section-title {
        color: #f1f3f5;
        font-size: 0.98rem;
        font-weight: 600;
        margin-bottom: 0.7rem;
    }

    .document-info {
        background-color: #131720;
        border: 1px solid #292f3a;
        border-radius: 12px;
        padding: 0.95rem;
        margin-bottom: 0.9rem;
    }

    .document-item {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .document-item + .document-item {
        margin-top: 0.85rem;
        padding-top: 0.75rem;
        border-top: 1px solid #292f3a;
    }

    .document-label {
        color: #8f98a8;
        font-size: 0.78rem;
        font-weight: 500;
    }

    .document-value {
        color: #f1f3f5;
        font-size: 0.9rem;
        font-weight: 500;
        word-break: break-word;
    }


    /* ========================================================
       DOCUMENT STATUS CARD
       ======================================================== */

    .status-card {
        background-color: #131720;
        border: 1px solid #292f3a;
        border-radius: 12px;
        padding: 0.95rem;
        margin-bottom: 0.9rem;
    }

    .status-title {
        color: #f1f3f5;
        font-size: 0.98rem;
        font-weight: 600;
        margin-bottom: 0.7rem;
    }

    .status-item {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        color: #c9ced6;
        font-size: 0.87rem;
        padding: 0.38rem 0;
    }

    .status-icon {
        color: #4ade80;
        font-weight: 700;
        font-size: 0.95rem;
        min-width: 16px;
    }


    /* ========================================================
       SUCCESS MESSAGE
       ======================================================== */

    [data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid #285f46;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border: none;
        border-top: 1px solid #292f3a;
        margin: 1.25rem 0;
    }


    /* ========================================================
       SIDEBAR BUTTON
       ======================================================== */

    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 9px;
        border: 1px solid #343a46;
        background-color: transparent;
        color: #d7dbe2;
        transition: all 0.2s ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: #596273;
        background-color: #20252e;
        color: #ffffff;
    }


    /* ========================================================
       CHAT MESSAGES
       ======================================================== */

    [data-testid="stChatMessage"] {
        border-radius: 12px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }

    /* User message */

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        background-color: #171a21;
        border: 1px solid #292f3a;
    }

    /* Assistant message */

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-assistant"]
    ) {
        background-color: #101319;
        border: 1px solid #232833;
    }

    [data-testid="stChatMessage"] p {
        line-height: 1.6;
        font-size: 0.98rem;
    }

    [data-testid="stChatMessage"] ul,
    [data-testid="stChatMessage"] ol {
        margin-top: 0.4rem;
        margin-bottom: 0.4rem;
    }


    /* ========================================================
       CHAT INPUT
       ======================================================== */

    [data-testid="stChatInput"] {
        background-color: #171a21;
        border: 1px solid #292f3a !important;
        border-radius: 12px;
        box-shadow: none !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border: 1px solid #596273 !important;
        box-shadow: none !important;
        outline: none !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #f1f3f5;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #8f98a8;
    }

    [data-testid="stChatInput"] textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }


    /* ========================================================
       GENERAL TEXT
       ======================================================== */

    p {
        color: #d7dbe2;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">✦ RAG-Based AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">An AI assistant for general questions and document-based Q&A.</div>',
    unsafe_allow_html=True
)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "chatbot" not in st.session_state:
    st.session_state.chatbot = RAGChatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )


    # ========================================================
    # LOAD PDF
    # ========================================================

    if uploaded_file is not None:

        if not st.session_state.pdf_loaded:

            with open("uploaded.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing PDF..."):
                st.session_state.chatbot = RAGChatbot("uploaded.pdf")

            st.session_state.pdf_loaded = True



        # ====================================================
        # DOCUMENT INFORMATION
        # ====================================================

        st.markdown("---")

        st.subheader("📄 Document Information")

        st.markdown(
            f"""
            **File:** `{uploaded_file.name}`

            **Size:** `{uploaded_file.size / 1024:.1f} KB`
            """
        )


        # ====================================================
        # PROCESSING SUCCESS
        # ====================================================

        if st.session_state.chatbot is not None:
            st.success("✓ Document processed successfully!")


        # ====================================================
        # DOCUMENT STATUS
        # ====================================================

        if st.session_state.chatbot is not None:

            st.markdown("---")

            st.subheader("📊 Document Status")

            st.success("✓ PDF processed")
            st.success("✓ Embeddings generated")
            st.success("✓ Vector store ready")


        # ====================================================
        # CLEAR CHAT
        # ====================================================

        st.markdown("---")

        if st.button(
            "🗑 Clear Chat",
            use_container_width=True
        ):

            st.session_state.messages = []

            if st.session_state.chatbot is not None:
                st.session_state.chatbot.history = []

            save_memory([])

            st.rerun()


# ============================================================
# DISPLAY PREVIOUS CHAT MESSAGES
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask me anything..."
)

if query:
    

        # ----------------------------------------------------
        # Display user message
        # ----------------------------------------------------

    with st.chat_message("user"):
        st.markdown(query)


        # ----------------------------------------------------
        # Store user message in Streamlit session
        # ----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })


        # ----------------------------------------------------
        # Generate assistant response
        # ----------------------------------------------------

    with st.chat_message("assistant"):

        try:

            with st.spinner("Thinking..."):
                answer = st.session_state.chatbot.chat(query)

            st.markdown(answer)

                # Store assistant response
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            print(f"ERROR: {type(e).__name__} → {e}")

            if "429" in str(e):
                answer = "⚠️ Gemini API quota has been reached. Please try again later."

            elif "503" in str(e):
                answer = "⚠️ Gemini is currently experiencing high demand. Please try again later."

            else:
                answer = "⚠️ Unable to generate a response right now. Please try again later."

            st.error(answer)


