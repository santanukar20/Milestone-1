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
    if not answer:
        return AskResponse(
            answer="Sorry, something went wrong while generating the answer.",
            source=source_url,
            last_updated=last_updated or utc_now_iso(),
        )
    
    # Check for specific error messages from the RAG system
    answer_lower = answer.strip().lower()
    if answer_lower.startswith("sorry, the ai service is currently busy"):
        return AskResponse(
            answer=answer.strip(),  # Return the specific rate limit message
            source=source_url,
            last_updated=last_updated or utc_now_iso(),
        )
    
    if answer_lower.startswith("sorry"):
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
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            :root {
                --bg-primary: #343541;
                --bg-secondary: #444654;
                --bg-input: #40414f;
                --bg-message-user: #444654;
                --bg-message-bot: #343541;
                --text-primary: #ececf1;
                --text-secondary: #acacbe;
                --accent-color: #10a37f;
                --border-color: #555;
            }
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                background-color: var(--bg-primary);
                color: var(--text-primary);
                height: 100vh;
                display: flex;
                flex-direction: column;
            }
            
            .container {
                max-width: 800px;
                width: 100%;
                margin: 0 auto;
                height: 100%;
                display: flex;
                flex-direction: column;
                padding: 16px;
            }
            
            .header {
                text-align: center;
                padding: 32px 0;
            }
            
            .header h1 {
                font-size: 2rem;
                font-weight: 600;
                margin-bottom: 16px;
            }
            
            .suggestions {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 12px;
                margin-bottom: 32px;
            }
            
            .suggestion-card {
                background-color: var(--bg-secondary);
                border: 1px solid var(--border-color);
                border-radius: 8px;
                padding: 16px;
                cursor: pointer;
                transition: all 0.2s ease;
                text-align: left;
            }
            
            .suggestion-card:hover {
                background-color: #555766;
                transform: translateY(-2px);
            }
            
            .chat-container {
                flex: 1;
                overflow-y: auto;
                padding: 24px 0;
                display: flex;
                flex-direction: column;
                gap: 24px;
            }
            
            .message {
                display: flex;
                gap: 24px;
                padding: 16px 0;
                max-width: 100%;
            }
            
            .message-user {
                background-color: var(--bg-message-user);
            }
            
            .message-bot {
                background-color: var(--bg-message-bot);
            }
            
            .avatar {
                width: 30px;
                height: 30px;
                border-radius: 2px;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-shrink: 0;
                font-weight: bold;
            }
            
            .user-avatar {
                background-color: #5436da;
            }
            
            .bot-avatar {
                background-color: #10a37f;
            }
            
            .message-content {
                flex: 1;
                padding-right: 40px;
                line-height: 1.5;
                overflow-wrap: break-word;
            }
            
            .message-content pre {
                white-space: pre-wrap;
            }
            
            .input-container {
                width: 100%;
                max-width: 800px;
                margin: 0 auto;
                padding: 24px 0;
            }
            
            .input-box {
                position: relative;
                border: 1px solid var(--border-color);
                border-radius: 24px;
                background-color: var(--bg-input);
                display: flex;
                align-items: center;
                padding: 8px 12px;
            }
            
            .input-box button.icon {
                background: none;
                border: none;
                color: var(--text-secondary);
                cursor: pointer;
                padding: 8px;
                font-size: 1.2rem;
            }
            
            .input-box input {
                flex: 1;
                background: transparent;
                border: none;
                color: var(--text-primary);
                padding: 8px 0;
                font-size: 1rem;
                outline: none;
                resize: none;
                max-height: 200px;
            }
            
            .input-box input::placeholder {
                color: var(--text-secondary);
            }
            
            .send-button {
                background-color: var(--accent-color);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 12px;
                cursor: pointer;
                font-weight: bold;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .send-button:hover {
                background-color: #0d8a6a;
            }
            
            .send-button:disabled {
                background-color: #555;
                cursor: not-allowed;
            }
            
            .footer {
                text-align: center;
                padding: 16px 0;
                color: var(--text-secondary);
                font-size: 0.8rem;
            }
            
            .citation {
                margin-top: 8px;
                font-size: 0.85em;
                color: var(--text-secondary);
            }
            
            .citation a {
                color: var(--accent-color);
                text-decoration: none;
            }
            
            .citation a:hover {
                text-decoration: underline;
            }
            
            .last-updated {
                margin-top: 4px;
                font-size: 0.8em;
                color: var(--text-secondary);
            }
            
            @media (max-width: 768px) {
                .suggestions {
                    grid-template-columns: 1fr;
                }
                
                .container {
                    padding: 8px;
                }
                
                .message {
                    gap: 16px;
                    padding: 12px 0;
                }
                
                .message-content {
                    padding-right: 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>What mutual fund fact should we begin with?</h1>
                <div class="suggestions">
                    <div class="suggestion-card" onclick="setSuggestion('What is the expense ratio of HDFC Flexi Cap Fund?')">
                        <p>What is the expense ratio of HDFC Flexi Cap Fund?</p>
                    </div>
                    <div class="suggestion-card" onclick="setSuggestion('What is the minimum SIP amount for HDFC Mid Cap Fund?')">
                        <p>What is the minimum SIP amount for HDFC Mid Cap Fund?</p>
                    </div>
                    <div class="suggestion-card" onclick="setSuggestion('What is the exit load for HDFC Small Cap Fund?')">
                        <p>What is the exit load for HDFC Small Cap Fund?</p>
                    </div>
                    <div class="suggestion-card" onclick="setSuggestion('Who manages the HDFC Large Cap Fund?')">
                        <p>Who manages the HDFC Large Cap Fund?</p>
                    </div>
                </div>
            </div>
            
            <div id="chatContainer" class="chat-container" style="display: none;">
                <div id="chat"></div>
            </div>
            
            <div class="input-container">
                <div class="input-box">
                    <button class="icon">➕</button>
                    <input id="queryInput" type="text" placeholder="Ask any factual question about mutual fund schemes..." />
                    <button class="icon">🎤</button>
                    <button id="sendButton" class="send-button" onclick="sendMessage()">Send</button>
                </div>
            </div>
            
            <div class="footer">
                <p>Facts-only. No investment advice. Sources: AMC, SEBI, AMFI.</p>
            </div>
        </div>

        <script>
            // Show chat interface when user starts typing or clicks a suggestion
            document.getElementById('queryInput').addEventListener('input', function() {
                if (this.value.trim() !== '') {
                    document.querySelector('.header').style.display = 'none';
                    document.getElementById('chatContainer').style.display = 'flex';
                }
            });
            
            function setSuggestion(text) {
                document.getElementById('queryInput').value = text;
                document.querySelector('.header').style.display = 'none';
                document.getElementById('chatContainer').style.display = 'flex';
                sendMessage();
            }
            
            async function sendMessage() {
                const input = document.getElementById('queryInput');
                const text = input.value.trim();
                if (!text) return;
                
                addMessage('user', text);
                input.value = '';
                toggleInputDisabled(true);
                
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
                    addBotMessage(answer, data.source, data.last_updated);
                } catch (e) {
                    addMessage('bot', 'Network error: ' + e);
                } finally {
                    toggleInputDisabled(false);
                }
            }
            
            function addMessage(who, text) {
                const chat = document.getElementById('chat');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message message-${who}`;
                
                const avatarDiv = document.createElement('div');
                avatarDiv.className = `avatar ${who}-avatar`;
                avatarDiv.textContent = who === 'user' ? 'U' : 'A';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.textContent = text;
                
                messageDiv.appendChild(avatarDiv);
                messageDiv.appendChild(contentDiv);
                chat.appendChild(messageDiv);
                chat.scrollTop = chat.scrollHeight;
            }
            
            function addBotMessage(answer, source, lastUpdated) {
                const chat = document.getElementById('chat');
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message message-bot';
                
                const avatarDiv = document.createElement('div');
                avatarDiv.className = 'avatar bot-avatar';
                avatarDiv.textContent = 'A';
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                
                const answerPre = document.createElement('pre');
                answerPre.textContent = answer;
                contentDiv.appendChild(answerPre);
                
                if (source) {
                    const citation = document.createElement('div');
                    citation.className = 'citation';
                    citation.innerHTML = '<strong>Source:</strong> <a href="' + source + '" target="_blank" rel="noopener noreferrer">' + source + '</a>';
                    contentDiv.appendChild(citation);
                }
                
                if (lastUpdated) {
                    const updated = document.createElement('div');
                    updated.className = 'last-updated';
                    updated.textContent = 'Last updated from sources: ' + new Date(lastUpdated).toLocaleString();
                    contentDiv.appendChild(updated);
                }
                
                messageDiv.appendChild(avatarDiv);
                messageDiv.appendChild(contentDiv);
                chat.appendChild(messageDiv);
                chat.scrollTop = chat.scrollHeight;
            }
            
            function toggleInputDisabled(disabled) {
                const input = document.getElementById('queryInput');
                const sendButton = document.getElementById('sendButton');
                input.disabled = disabled;
                sendButton.disabled = disabled;
                sendButton.textContent = disabled ? '...' : 'Send';
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
