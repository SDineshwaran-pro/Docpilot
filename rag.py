"""Multi-agent RAG on Google Cloud: Google ADK agents + Gemini (AI Studio or Vertex AI).

Planner agent  -> routes each question
  |- retrieval_agent: calls the search_docs tool (embedding search) and answers with citations
  |- general_agent:   answers general questions directly
"""
import asyncio
import io
import os
import time
import uuid

import numpy as np
from google import genai
from google.genai import errors
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ToolContext
from google.genai import types
from pypdf import PdfReader

CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
APP_NAME = "doc_assistant"
USER_ID = "local-user"

# session_id -> VectorIndex (kept in memory; the tool looks its index up by session)
INDEXES: dict[str, "VectorIndex"] = {}


def _client() -> genai.Client:
    # Reads GOOGLE_API_KEY, or GOOGLE_GENAI_USE_VERTEXAI / GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION
    return genai.Client()


# ---------------------------------------------------------------- ingestion
def _chunk(text: str, size: int = 900, overlap: int = 150):
    text = " ".join(text.split())
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step) if text[i:i + size].strip()]


def load_files(uploaded_files):
    """Streamlit UploadedFile list (pdf/txt/md) -> list of {text, source, page}."""
    pieces = []
    for f in uploaded_files:
        if f.name.lower().endswith(".pdf"):
            reader = PdfReader(io.BytesIO(f.getvalue()))
            for n, page in enumerate(reader.pages, start=1):
                for c in _chunk(page.extract_text() or ""):
                    pieces.append({"text": c, "source": f.name, "page": n})
        else:
            for c in _chunk(f.getvalue().decode("utf-8", errors="ignore")):
                pieces.append({"text": c, "source": f.name, "page": "-"})
    return pieces


def _with_retry(fn, tries: int = 4):
    """Free-tier keys hit rate limits (HTTP 429) easily; wait and retry instead of failing."""
    for i in range(tries):
        try:
            return fn()
        except errors.APIError as e:
            if getattr(e, "code", None) in (429, 503) and i < tries - 1:
                time.sleep(4 * 2 ** i)  # 4s, 8s, 16s
                continue
            raise


def _embed(texts, task_type):
    client, vectors = _client(), []
    for i in range(0, len(texts), 50):  # stay under the per-request batch limit
        res = _with_retry(lambda: client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i:i + 50],
            config=types.EmbedContentConfig(task_type=task_type),
        ))
        vectors.extend(e.values for e in res.embeddings)
    return np.array(vectors, dtype="float32")


class VectorIndex:
    def __init__(self, pieces):
        self.pieces = pieces
        m = _embed([p["text"] for p in pieces], "RETRIEVAL_DOCUMENT")
        self.matrix = m / np.linalg.norm(m, axis=1, keepdims=True)

    def search(self, query: str, k: int = 5):
        q = _embed([query], "RETRIEVAL_QUERY")[0]
        q = q / np.linalg.norm(q)
        top = np.argsort(self.matrix @ q)[::-1][:k]
        return [self.pieces[i] for i in top]


# ---------------------------------------------------------------- multimodal
def _generate(parts):
    res = _with_retry(lambda: _client().models.generate_content(model=CHAT_MODEL, contents=parts))
    return (res.text or "").strip()


def transcribe_audio(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    return _generate([types.Part.from_bytes(data=audio_bytes, mime_type=mime),
                      "Transcribe this audio exactly. Return only the transcript."])


def describe_image(image_bytes: bytes, mime: str = "image/png") -> str:
    return _generate([types.Part.from_bytes(data=image_bytes, mime_type=mime),
                      "Extract all readable text and describe the key content of this image."])


# ---------------------------------------------------------------- agents
def search_docs(query: str, tool_context: ToolContext) -> dict:
    """Search the user's uploaded documents and return the most relevant passages.

    Args:
        query: what to look for, phrased as a short search query.
    """
    index = INDEXES.get(tool_context.session.id)
    if index is None:
        return {"status": "error", "message": "No documents have been indexed yet."}
    hits = index.search(query)
    tool_context.state["last_sources"] = sorted({f"{h['source']} (p{h['page']})" for h in hits})
    return {"status": "ok", "passages": [
        {"source": h["source"], "page": h["page"], "text": h["text"]} for h in hits]}


def build_agent() -> LlmAgent:
    retrieval_agent = LlmAgent(
        name="retrieval_agent",
        model=CHAT_MODEL,
        description="Answers questions that depend on the user's uploaded documents.",
        instruction=(
            "Always call search_docs first. Answer ONLY from the passages it returns. "
            "If the answer is not there, say you could not find it. "
            "Cite sources as [file, page]. Keep the answer under 6 sentences."
        ),
        tools=[search_docs],
    )
    general_agent = LlmAgent(
        name="general_agent",
        model=CHAT_MODEL,
        description="Answers general questions and small talk that do not need the documents.",
        instruction="Answer clearly and briefly.",
    )
    return LlmAgent(
        name="planner",
        model=CHAT_MODEL,
        description="Routes each question to the right agent.",
        instruction=(
            "You are a planner. If the user's message says documents are indexed and the question "
            "is about their content (or about an attached image or voice note), transfer to "
            "retrieval_agent. Otherwise transfer to general_agent. Do not answer yourself."
        ),
        sub_agents=[retrieval_agent, general_agent],
    )


class Assistant:
    """One chat session: ADK session + in-memory vector index."""

    def __init__(self):
        self.session_service = InMemorySessionService()
        self.session_id = uuid.uuid4().hex
        self.agent = build_agent()
        asyncio.run(self.session_service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=self.session_id))

    @property
    def has_index(self) -> bool:
        return self.session_id in INDEXES

    def build_index(self, pieces) -> int:
        INDEXES[self.session_id] = VectorIndex(pieces)
        return len(pieces)

    async def _ask(self, text: str):
        runner = Runner(agent=self.agent, app_name=APP_NAME, session_service=self.session_service)
        msg = types.Content(role="user", parts=[types.Part(text=text)])
        final, author = "", ""
        async for ev in runner.run_async(user_id=USER_ID, session_id=self.session_id, new_message=msg):
            if ev.is_final_response() and ev.content and ev.content.parts:
                final = "".join(p.text or "" for p in ev.content.parts)
                author = ev.author
        session = await self.session_service.get_session(
            app_name=APP_NAME, user_id=USER_ID, session_id=self.session_id)
        sources = session.state.get("last_sources", []) if author == "retrieval_agent" else []
        return final, author, sources

    def ask(self, question: str, extra_context: str = ""):
        """Returns (answer, agent_that_answered, sources)."""
        parts = [f"[documents indexed: {'yes' if self.has_index else 'no'}]"]
        if extra_context:
            parts.append(f"[attached image content: {extra_context}]")
        parts.append(question)
        return asyncio.run(self._ask("\n".join(parts)))
