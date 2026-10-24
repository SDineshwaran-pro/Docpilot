import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# Streamlit Community Cloud keeps keys in st.secrets; locally we use .env.
# On Google Cloud (Cloud Run) set GOOGLE_GENAI_USE_VERTEXAI=TRUE and no key is needed.
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

import rag  # noqa: E402  (imported after the environment is set)

st.set_page_config(page_title="Multi-Agent Document Assistant", page_icon="📄", layout="wide")
st.title("📄 Multi-Agent Document Assistant")
st.caption("Google ADK agents + Gemini over your own files. Ask by text, image or voice.")

use_vertex = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() in ("1", "TRUE")
if not (os.getenv("GOOGLE_API_KEY") or use_vertex):
    st.error("Set GOOGLE_API_KEY (local / Streamlit Cloud) or GOOGLE_GENAI_USE_VERTEXAI=TRUE (Google Cloud).")
    st.stop()

if "assistant" not in st.session_state:
    st.session_state.assistant = rag.Assistant()
    st.session_state.history = []
assistant = st.session_state.assistant

# ---------- Sidebar: ingestion ----------
with st.sidebar:
    st.header("1. Add documents")
    files = st.file_uploader("PDF / TXT / MD", type=["pdf", "txt", "md"], accept_multiple_files=True)
    if st.button("Build index", disabled=not files):
        with st.spinner("Reading, chunking and embedding..."):
            n = assistant.build_index(rag.load_files(files))
        st.success(f"Indexed {len(files)} file(s) into {n} chunks.")
    st.divider()
    st.caption(f"Model: {rag.CHAT_MODEL} • {'Vertex AI' if use_vertex else 'Gemini API'}")
    if not use_vertex:
        st.warning("Free Gemini API tier: Google may use submitted content to improve its products. "
                   "Do not upload confidential or personal documents.")

# ---------- Main: ask ----------
for role, text in st.session_state.history:
    with st.chat_message(role):
        st.markdown(text)

col1, col2 = st.columns(2)
with col1:
    image = st.file_uploader("Optional: screenshot or scanned page", type=["png", "jpg", "jpeg"], key="img")
with col2:
    voice = st.audio_input("Optional: ask by voice")

question = st.chat_input("Ask about your documents...")

extra = ""
if image is not None:
    with st.spinner("Reading image..."):
        extra = rag.describe_image(image.getvalue(), image.type or "image/png")
if voice is not None and not question:
    with st.spinner("Transcribing voice note..."):
        question = rag.transcribe_audio(voice.getvalue())

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        start = time.time()
        try:
            with st.spinner("Agents working..."):
                text, agent_name, sources = assistant.ask(question, extra)
        except Exception as e:  # most often the free-tier rate limit
            st.error("The Gemini free tier is busy or rate-limited. Wait a minute and try again.")
            st.caption(str(e)[:300])
            st.stop()
        st.markdown(text)
        st.caption(f"Answered by: {agent_name} • {time.time() - start:.1f}s"
                   + (f" • Sources: {', '.join(sources)}" if sources else ""))
    st.session_state.history.append(("assistant", text))
