import os
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔍",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0f1117;
        color: #e2e8f0;
    }

    .header {
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid #1e2530;
        margin-bottom: 2rem;
    }

    .header h1 {
        font-size: 1.6rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .header p {
        color: #64748b;
        font-size: 0.85rem;
        margin: 0.3rem 0 0 0;
    }

    .answer-box {
        background: #151b27;
        border: 1px solid #1e293b;
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #cbd5e1;
    }

    .source-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #94a3b8;
        line-height: 1.6;
    }

    .source-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }

    .score-badge {
        display: inline-block;
        background: #1e293b;
        color: #94a3b8;
        font-size: 0.68rem;
        padding: 2px 8px;
        border-radius: 20px;
        margin-left: 8px;
        font-family: 'JetBrains Mono', monospace;
    }

    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        background: #22c55e;
        border-radius: 50%;
        margin-right: 6px;
    }

    .status-text {
        font-size: 0.75rem;
        color: #64748b;
    }

    div[data-testid="stTextInput"] input {
        background: #151b27 !important;
        border: 1px solid #1e293b !important;
        color: #e2e8f0 !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
    }

    div[data-testid="stTextInput"] input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
    }

    .stButton > button {
        background: #6366f1 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        padding: 0.5rem 1.5rem !important;
        transition: background 0.15s !important;
    }

    .stButton > button:hover {
        background: #4f46e5 !important;
    }

    .sidebar-section {
        background: #151b27;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 0.875rem 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.8rem;
        color: #94a3b8;
    }

    .sidebar-section strong {
        color: #e2e8f0;
        display: block;
        margin-bottom: 0.3rem;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(BASE_DIR, "db", "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen2.5:1.5b"

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )
    return db

@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)

# ── Core RAG function ─────────────────────────────────────────────────────────
def run_rag(query: str, k: int, score_threshold: float):
    db = load_vectorstore()
    llm = load_llm()

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold},
    )

    docs = retriever.invoke(query)

    if not docs:
        return None, []

    context = "\n\n".join([f"[Doc {i+1}]\n{d.page_content}" for i, d in enumerate(docs)])

    prompt = f"""Answer the following question using ONLY the documents provided below.
If the answer is not found in the documents, say: "I don't have enough information in the loaded documents to answer this."

Question: {query}

Documents:
{context}

Answer:"""

    messages = [
        SystemMessage(content="You are a precise research assistant. Answer only from the given documents."),
        HumanMessage(content=prompt),
    ]

    result = llm.invoke(messages)
    return result.content, docs

# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <h1>🔍 RAG Research Assistant</h1>
    <p>Retrieval-Augmented Generation · LangChain · ChromaDB · HuggingFace · Ollama</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### Settings")
    k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=10, value=5)
    score_threshold = st.slider("Similarity threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""<div class="sidebar-section">
        <strong>Embedding Model</strong>
        all-MiniLM-L6-v2
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="sidebar-section">
        <strong>LLM</strong>
        {OLLAMA_MODEL} via Ollama
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="sidebar-section">
        <strong>Vector Store</strong>
        ChromaDB · cosine similarity
    </div>""", unsafe_allow_html=True)

    # Check Ollama status
    try:
        load_llm()
        st.markdown('<p class="status-text"><span class="status-dot"></span>Ollama running</p>', unsafe_allow_html=True)
    except Exception:
        st.markdown('<p class="status-text" style="color:#ef4444;">⚠ Ollama not detected — start it locally</p>', unsafe_allow_html=True)

# Main area
col1, col2 = st.columns([5, 1])
with col1:
    query = st.text_input("", placeholder="Ask a question about your documents...")
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("Search", use_container_width=True)

if search_btn and query.strip():
    with st.spinner("Retrieving & generating..."):
        answer, docs = run_rag(query, k, score_threshold)

    if answer is None:
        st.warning("No relevant chunks found. Try lowering the similarity threshold or rephrasing your query.")
    else:
        st.markdown("#### Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        st.markdown(f"#### Source Chunks <span style='color:#64748b;font-size:0.8rem;font-weight:400;'>({len(docs)} retrieved)</span>", unsafe_allow_html=True)
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            filename = os.path.basename(source) if source != "unknown" else "unknown"
            st.markdown(f"""
            <div class="source-card">
                <div class="source-label">Chunk {i+1} · {filename}</div>
                {doc.page_content}
            </div>
            """, unsafe_allow_html=True)

elif search_btn and not query.strip():
    st.warning("Enter a question first.")

# Empty state
if not (search_btn and query.strip()):
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #334155;">
        <div style="font-size:2.5rem; margin-bottom:0.75rem;">🗂️</div>
        <div style="font-size:0.9rem;">Enter a question above to query your document store</div>
    </div>
    """, unsafe_allow_html=True)