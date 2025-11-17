# Mutual Fund FAQ Chatbot - Milestone 1

## Project Summary

This project implements a Retrieval-Augmented Generation (RAG) chatbot for answering mutual fund FAQs. The system extracts structured data from HDFC Mutual Fund website, processes it into searchable chunks, generates embeddings using Google Gemini API, and provides accurate answers to user questions.

## Components

### 1. Data Extraction (`scraper.py`)
- Scrapes 5 HDFC mutual fund schemes:
  - HDFC Flexi Cap Fund
  - HDFC Balanced Advantage Fund
  - HDFC Large Cap Fund
  - HDFC Mid Cap Fund
  - HDFC Small Cap Fund
- Extracts key information:
  - Fund name, category, risk level
  - AUM (Assets Under Management)
  - Expense Ratio (TER)
  - Minimum SIP amount
  - Exit load information
  - Returns since inception
  - Fund managers

### 2. Data Processing (`chunker.py`)
- Converts structured data into 30 searchable chunks (6 per scheme)
- Each chunk contains specific information about the fund

### 3. Embedding Generation (`embedder_gemini_simple.py`)
- Generates 768-dimensional embeddings for all chunks using Google Gemini API
- Saves embeddings to `data/embeddings.json`

### 4. RAG Backend (`rag_backend_simple.py`)
- Implements cosine similarity search to find relevant chunks
- Uses Google Gemini to generate natural language answers
- Interactive chat interface

## Key Features

- **Accurate Information**: All data sourced directly from HDFC Mutual Fund website
- **Fast Response**: Embedding-based search for quick retrieval
- **Citation Support**: Each answer can include source URLs
- **Extensible**: Easy to add more funds or data sources

## How It Works

1. **Data Extraction**: Web scraper collects structured data from HDFC website
2. **Chunking**: Data is organized into logical chunks for efficient retrieval
3. **Embedding**: Each chunk is converted to a numerical vector using Gemini
4. **Search**: User questions are matched to relevant chunks using cosine similarity
5. **Generation**: Gemini generates natural language answers using retrieved context

## Files Generated

- `data/raw_scheme_pages.jsonl` - Raw scraped data (5 records)
- `data/processed_chunks.jsonl` - Structured chunks (30 chunks)
- `data/embeddings.json` - Vector embeddings and metadata

## Usage

To run the chatbot:
```bash
python rag_backend_simple.py
```

Then ask questions like:
- "What is the expense ratio of HDFC Flexi Cap Fund?"
- "What is the minimum SIP amount for HDFC Mid Cap Fund?"
- "What is the exit load for HDFC Small Cap Fund?"

## Technical Stack

- **Python** - Core programming language
- **BeautifulSoup** - Web scraping
- **Google Gemini API** - Embeddings and text generation
- **NumPy** - Numerical computations
- **JSON** - Data storage format

## Success Metrics

- ✅ 5/5 HDFC schemes successfully scraped
- ✅ 100% extraction of key fields (AUM, TER, SIP minimum, etc.)
- ✅ 30 high-quality chunks generated
- ✅ 30 embeddings successfully created
- ✅ Semantic search working correctly
- ✅ Natural language answers generated

This completes Milestone 1 with a fully functional RAG pipeline ready for deployment.