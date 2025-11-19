# Mutual Fund Facts-Only Assistant

A lightweight Retrieval-Augmented Generation (RAG) chatbot that answers factual questions about mutual funds using data sourced from official AMC/SEBI/AMFI pages. The system extracts structured data, generates embeddings, and provides accurate, citation-backed answers through a FastAPI backend with an interactive web interface.

## Features

- **Factual Accuracy**: All information sourced directly from official mutual fund websites
- **Fast Responses**: Embedding-based semantic search for quick retrieval
- **Citation Support**: Each answer includes source URLs and last-updated timestamps
- **No Investment Advice**: Strictly facts-only, no recommendations or opinions
- **Extensible Design**: Easy to add more funds or data sources

## Tech Stack

- **Python** - Core programming language
- **FastAPI** - High-performance web framework
- **Google Gemini API** - Embeddings and text generation
- **BeautifulSoup** - Web scraping
- **NumPy** - Numerical computations
- **Streamlit** - Alternative frontend option

## Project Structure

```
├── app.py                 # FastAPI backend with /ask endpoint
├── rag_backend_simple.py  # RAG implementation with Gemini
├── scraper.py             # Web scraper for mutual fund data
├── chunker.py             # Data processing and chunking
├── embedder_gemini_simple.py # Embedding generation
├── data/
│   ├── embeddings.json    # Pre-generated vector embeddings
│   ├── processed_chunks.jsonl # Structured data chunks
│   └── raw_scheme_pages.jsonl # Raw scraped data
└── requirements.txt       # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Create a virtual environment:
```bash
python -m venv .venv
```

3. Activate the virtual environment:
```bash
# On Windows
.venv\Scripts\activate

# On Linux/Mac
source .venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: FastAPI Backend with Web UI (Recommended)

Start the FastAPI server:
```bash
python app.py
```

Access the chat interface at:
```
http://127.0.0.1:8000/
```

### Option 2: Streamlit Frontend

```bash
streamlit run frontend_app.py
```

### Option 3: Direct RAG Backend

```bash
python rag_backend_simple.py
```

## API Usage

The FastAPI backend provides the following endpoints:

- `POST /ask` - Ask a mutual fund question
  ```bash
  curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "What is the expense ratio of HDFC Flexi Cap Fund?"}'
  ```

- `GET /health` - Health check endpoint
- `GET /docs` - Interactive Swagger UI for API testing

## How the RAG System Works

1. **Data Extraction**: Web scraper collects structured data from mutual fund websites
2. **Chunking**: Data is organized into logical chunks for efficient retrieval
3. **Embedding**: Each chunk is converted to a numerical vector using Google Gemini API
4. **Storage**: Embeddings and metadata are stored in `data/embeddings.json`
5. **Search**: User questions are matched to relevant chunks using cosine similarity
6. **Generation**: Gemini generates natural language answers using retrieved context

## Data Sources

The chatbot currently uses data scraped from the following HDFC Mutual Fund pages:

- [HDFC Flexi Cap Fund](https://www.hdfcfund.com/explore/mutual-funds/hdfc-flexi-cap-fund/direct)
- [HDFC Balanced Advantage Fund](https://www.hdfcfund.com/explore/mutual-funds/hdfc-balanced-advantage-fund/direct)
- [HDFC Large Cap Fund](https://www.hdfcfund.com/explore/mutual-funds/hdfc-large-cap-fund/direct)
- [HDFC Mid Cap Fund](https://www.hdfcfund.com/explore/mutual-funds/hdfc-mid-cap-fund/direct)
- [HDFC Small Cap Fund](https://www.hdfcfund.com/explore/mutual-funds/hdfc-small-cap-fund/direct)

## Updating Documents/Embeddings

To update the mutual fund data:

1. Run the scraper to collect fresh data:
```bash
python scraper.py
```

2. Process the data into chunks:
```bash
python chunker.py
```

3. Generate new embeddings:
```bash
python embedder_gemini_simple.py
```

The system will automatically use the updated embeddings on next restart.

## Free API Options

### Google Gemini (Recommended)
- Get a free API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- Set the environment variable:
  ```bash
  export GEMINI_API_KEY=your-api-key-here
  ```

### Hugging Face (Alternative)
- No API key required for basic usage
- For higher rate limits, get a free token at [Hugging Face](https://huggingface.co/settings/tokens)

## Deployment

The application is currently deployed on Render and can be accessed at: [https://mf-facts-bot.onrender.com/](https://mf-facts-bot.onrender.com/)

### Render Deployment Configuration

1. **Build Command**: `pip install -r requirements.txt`
2. **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
3. **Environment Variables**: Set `GEMINI_API_KEY` in Render dashboard
4. **Region**: Choose a region closest to your users
5. **Auto-Deploy**: Enable auto-deployment from GitHub for automatic updates

### Accessing the Deployed Application

Once deployed, you can access:
- **Chat Interface**: [https://mf-facts-bot.onrender.com/](https://mf-facts-bot.onrender.com/)
- **API Documentation**: [https://mf-facts-bot.onrender.com/docs](https://mf-facts-bot.onrender.com/docs)
- **Health Check**: [https://mf-facts-bot.onrender.com/health](https://mf-facts-bot.onrender.com/health)

### Local Development vs Production

For local development, use:
```bash
python app.py
```

For production deployment on Render, the platform automatically uses:
```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

The `$PORT` environment variable is automatically provided by Render.

## Environment Variables

- `GEMINI_API_KEY` - Google Gemini API key (required for embeddings and generation)
- `HUGGINGFACE_API_TOKEN` - Hugging Face token (optional, for higher rate limits)

## Example Questions

- "What is the expense ratio of HDFC Flexi Cap Fund?"
- "What is the minimum SIP amount for HDFC Mid Cap Fund?"
- "What is the exit load for HDFC Small Cap Fund?"
- "Who manages the HDFC Large Cap Fund?"
- "What is the risk level of HDFC Balanced Advantage Fund?"

## Limitations

- Rate limits on free API tiers may affect response times
- Data is limited to the information available on scraped websites
- NAV data may not be available on main fund pages (website limitation, not system limitation)

## Future Improvements

1. **Expand Data Sources**: Add support for more mutual fund companies beyond HDFC
2. **Real-time NAV Updates**: Implement direct API integration for real-time NAV data
3. **Multi-language Support**: Add support for regional languages to reach wider audience
4. **Enhanced UI/UX**: Improve the chat interface with better styling and user experience
5. **Voice Input**: Add voice-to-text capability for hands-free querying
6. **Caching Mechanism**: Implement caching for frequently asked questions to reduce API usage
7. **Advanced Analytics**: Add usage analytics to understand common user queries
8. **Mobile App**: Develop a dedicated mobile application for better accessibility
9. **Personalization**: Add user profiles to track frequently asked funds
10. **Offline Mode**: Implement offline functionality for basic queries
