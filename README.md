# Mutual Fund FAQ RAG Chatbot

Production-grade RAG chatbot for Mutual Fund FAQs using HDFC AMC data.

## Components

1. **scraper.py** - Scrapes scheme pages from HDFC AMC website
2. **chunker.py** - Splits scraped data into RAG-optimized chunks
3. **embedder.py** - Generates embeddings and builds FAISS index
4. **rag_backend.py** - FastAPI backend with RAG pipeline

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### Step 1: Scrape Data
```bash
python scraper.py
```
Creates: `data/raw_scheme_pages.jsonl`

### Step 2: Chunk Data
```bash
python chunker.py
```
Creates: `data/processed_chunks.jsonl`

### Step 3: Generate Embeddings
```bash
python embedder.py
```
Creates: 
- `data/faiss_index.bin`
- `data/metadata.jsonl`

### Step 4: Start Backend
```bash
python rag_backend.py
```
Or with uvicorn:
```bash
uvicorn rag_backend:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### GET /health
Returns server health status.

**Response:**
```json
{
  "status": "ok"
}
```

### POST /ask
Answer a question using RAG pipeline.

**Request:**
```json
{
  "question": "What is the investment objective of HDFC Flexi Cap Fund?"
}
```

**Response:**
```json
{
  "mode": "FACT",
  "answer": "The HDFC Flexi Cap Fund aims to generate capital appreciation...",
  "citation_url": "https://www.hdfcfund.com/explore/..."
}
```

**Response Modes:**
- `FACT` - Factual answer provided
- `REFUSAL` - Question asks for forbidden content (advice/recommendations)
- `NO_DATA` - No relevant information found (similarity < 0.75)
- `ERROR` - Error occurred

## Response Modes

- **FACT**: Factual answer based on retrieved context
- **REFUSAL**: Question asks for buy/sell recommendations, portfolio advice, or predictions
- **NO_DATA**: Similarity score < 0.75, insufficient relevant information
- **ERROR**: System error occurred

## Safety Features

- No buy/sell recommendations
- No portfolio advice
- No predictions about future performance
- Factual answers only (≤3 sentences)
- Context-only responses (no external knowledge)
- Strict similarity threshold (0.75)
