# Mutual Fund Facts-Only Assistant

Lightweight Retrieval-Augmented Generation (RAG) assistant focused on factual mutual fund FAQs (expense ratios, SIP minimums, exit loads, etc.) sourced from official AMC / SEBI / AMFI pages. The backend exposes a FastAPI `/ask` endpoint and serves a minimal browser UI for quick Q&A with citations.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # PowerShell / cmd on Windows
pip install -r requirements.txt
```

Ensure you have the pre-generated embeddings under `data/` (already included for Milestone‑1).

## Running the backend

```bash
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --port 8000
```

The server provides:

- `POST /ask` – returns a concise factual answer, source URL, and last-updated timestamp  
- `GET /health` – lightweight readiness probe  
- `GET /docs` – interactive Swagger UI for manual testing

## Using the frontend

The FastAPI service also serves a built-in chat page. After the backend is running, open:

```
http://127.0.0.1:8000/
```

Type one of the suggested prompts (e.g., “What is the expense ratio of HDFC Flexi Cap?”) to see a citation-backed response in real time. If you prefer using the API directly, you can hit `http://127.0.0.1:8000/docs` and invoke `/ask` from the Swagger interface.
