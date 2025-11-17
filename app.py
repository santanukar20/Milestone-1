"""
FastAPI wrapper around the lightweight Mutual Fund RAG backend.
Provides a simple /ask endpoint for the frontend chat UI.
"""

from fastapi.responses import HTMLResponse
from datetime import datetime, timezone
from typing import List, Optional, Tuple
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_backend_simple import MutualFundRAG


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    source: Optional[str]
    last_updated: str


app = FastAPI(
    title="Mutual Fund Facts-Only API",
    description="Facts-only RAG API for Mutual Fund FAQ assistant",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system: Optional[MutualFundRAG] = None

FORBIDDEN_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bshould\b",
        r"\binvest\b",
        r"\bbuy\b",
        r"\bsell\b",
        r"\brecommend\b",
        r"\bsuggestion\b",
        r"\bportfolio\b",
        r"\bwhich fund\b",
        r"\bwhat to\b",
        r"\bbetter fund\b",
        r"\bforecast\b",
        r"\bprediction\b",
    ]
]


def utc_now_iso() -> str:
    """Return ISO 8601 timestamp in UTC with trailing Z."""
    return datetime.now(timezone.utc).isoformat()


def question_requests_advice(question: str) -> bool:
    """Detect if a question is asking for advice or opinions."""
    q_lower = question.lower()
    return any(pattern.search(q_lower) for pattern in FORBIDDEN_PATTERNS)


def extract_context(
    indices: List[Tuple[int, float]]
) -> Tuple[List[str], Optional[str], Optional[str]]:
    """
    Build context chunks plus citation data for retrieved indices.

    Returns:
        context_chunks, source_url, last_updated
    """
    assert rag_system is not None
    context_chunks: List[str] = []
    source_url: Optional[str] = None
    last_updated: Optional[str] = None

    for idx, _ in indices:
        if idx < 0 or idx >= len(rag_system.documents):
            continue
        context_chunks.append(rag_system.documents[idx])
        metadata = rag_system.metadata[idx]
        if source_url is None:
            source_url = metadata.get("source_url")
        if last_updated is None:
            last_updated = metadata.get("last_scraped_at")

    return context_chunks, source_url, last_updated


@app.on_event("startup")
async def startup_event():
    """Load embeddings and prepare the RAG system."""
    global rag_system
    rag_system = MutualFundRAG()


@app.get("/health")
async def health() -> dict:
    """Simple health check."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """Answer a mutual fund factual question."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG backend not initialized.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if question_requests_advice(question):
        return AskResponse(
            answer="Facts-only. No investment advice. "
            "Please ask about specific factual items like expense ratios, SIP minimums, exit loads, or riskometers.",
            source=None,
            last_updated=utc_now_iso(),
        )

    relevant = rag_system.find_relevant_chunks(question, top_k=3)
    if not relevant:
        return AskResponse(
            answer="I could not find relevant official data for that question. "
            "Please try rephrasing or ask about another scheme detail.",
            source=None,
            last_updated=utc_now_iso(),
        )

    context_chunks, source_url, last_updated = extract_context(relevant)

    if not context_chunks:
        return AskResponse(
            answer="I could not locate enough context for that question. "
            "Please try a different fact-focused query.",
            source=None,
            last_updated=utc_now_iso(),
        )

    answer = rag_system.generate_answer(question, context_chunks)
    if not answer or answer.strip().lower().startswith("sorry"):
        return AskResponse(
            answer="Sorry, something went wrong while generating the answer.",
            source=source_url,
            last_updated=last_updated or utc_now_iso(),
        )

    return AskResponse(
        answer=answer.strip(),
        source=source_url,
        last_updated=last_updated or utc_now_iso(),
    )
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mutual Fund Facts-Only Chatbot</title>
        <meta charset="utf-8" />
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 40px auto; }
            h1 { margin-bottom: 0.2rem; }
            small { color: #666; }
            #chat { border: 1px solid #ddd; padding: 10px; height: 400px; overflow-y: auto; margin: 16px 0; }
            .msg-user { text-align: right; margin: 4px 0; }
            .msg-bot { text-align: left; margin: 4px 0; }
            .bubble-user { display: inline-block; background: #007bff; color: #fff; padding: 6px 10px; border-radius: 8px; }
            .bubble-bot { display: inline-block; background: #f5f5f5; padding: 6px 10px; border-radius: 8px; }
            #inputRow { display: flex; gap: 8px; }
            #queryInput { flex: 1; padding: 6px; }
            button { padding: 6px 16px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>Mutual Fund Facts-Only Assistant</h1>
        <small>Facts-only. No investment advice.</small>

        <div id="chat"></div>

        <div id="inputRow">
            <input id="queryInput" type="text" placeholder="Ask about a mutual fund scheme..." />
            <button onclick="sendMessage()">Send</button>
        </div>

        <script>
            async function sendMessage() {
                const input = document.getElementById('queryInput');
                const text = input.value.trim();
                if (!text) return;

                addMessage('user', text);
                input.value = '';

                try {
                    const resp = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: text })
                    });

                    if (!resp.ok) {
                        addMessage('bot', 'Server error: ' + resp.status);
                        return;
                    }

                    const data = await resp.json();
                    const answer = data.answer || JSON.stringify(data);
                    addMessage('bot', answer);
                } catch (e) {
                    addMessage('bot', 'Network error: ' + e);
                }
            }

            function addMessage(who, text) {
                const chat = document.getElementById('chat');
                const wrap = document.createElement('div');
                wrap.className = who === 'user' ? 'msg-user' : 'msg-bot';

                const bubble = document.createElement('div');
                bubble.className = who === 'user' ? 'bubble-user' : 'bubble-bot';
                bubble.textContent = text;

                wrap.appendChild(bubble);
                chat.appendChild(wrap);
                chat.scrollTop = chat.scrollHeight;
            }

            document.getElementById('queryInput').addEventListener('keydown', function(e) {
                if (e.key === 'Enter') sendMessage();
            });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
