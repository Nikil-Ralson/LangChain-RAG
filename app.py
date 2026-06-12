import streamlit as st
import requests
import uuid
import os
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind — RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0d0d0f;
    --surface: #16161a;
    --border: #2a2a30;
    --accent: #7fff6e;
    --accent-dim: #4aad43;
    --text: #e8e8ec;
    --text-muted: #6b6b78;
    --user-bubble: #1e2a1e;
    --ai-bubble: #16161a;
}

html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important; font-family: 'DM Mono', monospace; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

[data-testid="stFileUploader"] { background: var(--surface); border: 1px dashed var(--border); border-radius: 8px; padding: 8px; }

.chat-message { padding: 14px 18px; border-radius: 10px; margin-bottom: 12px; line-height: 1.65; font-size: 0.9rem; border: 1px solid var(--border); }
.chat-message.user { background: var(--user-bubble); border-left: 3px solid var(--accent); }
.chat-message.assistant { background: var(--ai-bubble); border-left: 3px solid var(--text-muted); }
.role-label { font-family: 'Syne', sans-serif; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; }
.role-label.user { color: var(--accent); }
.role-label.assistant { color: var(--text-muted); }

.source-tag { display: inline-block; background: #1e1e28; border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; color: var(--text-muted); margin-right: 6px; margin-top: 8px; font-family: 'DM Mono', monospace; }

.stTextInput > div > div > input { background: var(--surface) !important; border: 1px solid var(--border) !important; color: var(--text) !important; border-radius: 8px !important; font-family: 'DM Mono', monospace !important; }
.stTextInput > div > div > input:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(127,255,110,0.15) !important; }

.stButton > button { background: var(--accent) !important; color: #0d0d0f !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; border: none !important; border-radius: 6px !important; letter-spacing: 0.05em; }
.stButton > button:hover { background: var(--accent-dim) !important; }

.stSelectbox > div > div { background: var(--surface) !important; border-color: var(--border) !important; color: var(--text) !important; }
.stAlert { border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "doc_map" not in st.session_state:
    st.session_state.doc_map = {}   # {doc_id: filename}
if "selected_doc_id" not in st.session_state:
    st.session_state.selected_doc_id = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind")
    st.markdown("<span style='color:#6b6b78;font-size:0.8rem'>RAG · Gemini · pgvector</span>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "PDF, TXT, or MD",
        type=["pdf", "txt", "md"],
        label_visibility="collapsed",
    )
    tags_input = st.text_input(
        "Tags",
        placeholder="e.g. tax, 2024, finance",
        help="Comma-separated tags to categorize this document",
    )

    if uploaded_file:
        # Check by filename — re-upload same file is a no-op
        already_uploaded = uploaded_file.name in st.session_state.doc_map.values()
        if not already_uploaded:
            with st.spinner("Ingesting document..."):
                res = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"tags": tags_input},
                )
            if res.status_code == 200:
                data = res.json()
                st.session_state.doc_map[data["doc_id"]] = uploaded_file.name
                tag_str = ", ".join(data.get("tags", [])) or "none"
                st.success(f"✓ {data['message']}\nTags: {tag_str}")
            else:
                st.error(f"Upload failed: {res.json().get('detail', 'Unknown error')}")

    # ── Document selector ─────────────────────────────────────────────────────
    if st.session_state.doc_map:
        st.divider()
        st.markdown("### Chat Scope")

        options = {"All documents": None}
        options.update({name: doc_id for doc_id, name in st.session_state.doc_map.items()})

        selected_label = st.selectbox(
            "Search within",
            options=list(options.keys()),
            label_visibility="collapsed",
        )
        st.session_state.selected_doc_id = options[selected_label]

        if st.session_state.selected_doc_id is None:
            st.caption("🔍 Searching across all uploaded documents")
        else:
            st.caption(f"🔍 Searching only in: **{selected_label}**")

        st.divider()
        st.markdown("### Uploaded Documents")
        for doc_id, name in st.session_state.doc_map.items():
            st.markdown(f"<div class='source-tag'>📄 {name}</div>", unsafe_allow_html=True)

    st.divider()
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.markdown(
        f"<div style='color:#2a2a30;font-size:0.65rem;margin-top:24px'>session: {st.session_state.session_id[:8]}…</div>",
        unsafe_allow_html=True,
    )


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-family:Syne,sans-serif;font-weight:800;font-size:2rem;margin-bottom:0'>Document Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#6b6b78;font-size:0.85rem;margin-top:4px'>Upload documents → ask anything about them</p>", unsafe_allow_html=True)
st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center;padding:48px 0;color:#2a2a30'>
        <div style='font-size:2.5rem'>📂</div>
        <div style='font-family:Syne,sans-serif;font-size:1rem;margin-top:12px'>
            Upload a document and start asking questions
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])
        label = "YOU" if role == "user" else "DOCMIND"
        sources_html = ""
        if sources:
            tags = "".join(f"<span class='source-tag'>📄 {s}</span>" for s in sources)
            sources_html = f"<div>{tags}</div>"

        st.markdown(f"""
        <div class='chat-message {role}'>
            <div class='role-label {role}'>{label}</div>
            <div>{content}</div>
            {sources_html}
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_input(
        "Ask a question",
        placeholder="What does this document say about…",
        label_visibility="collapsed",
        key="question_input",
    )
with col2:
    send = st.button("Send →", use_container_width=True)

if send and question.strip():
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking…"):
        try:
            res = requests.post(
                f"{API_URL}/chat",
                json={
                    "session_id": st.session_state.session_id,
                    "question": question,
                    "doc_id": st.session_state.selected_doc_id,  # None = all docs
                },
                timeout=30,
            )
            if res.status_code == 200:
                data = res.json()
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", []),
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠ Error: {res.json().get('detail', 'Something went wrong.')}",
                    "sources": [],
                })
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to the API. Is the FastAPI server running on port 8000?")

    st.rerun()
