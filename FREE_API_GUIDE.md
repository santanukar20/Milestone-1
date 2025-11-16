# Free API Setup Guide

This guide explains how to use **FREE** APIs instead of paid OpenAI services for the RAG chatbot.

## Free API Options

### Option 1: Hugging Face Inference API (Completely FREE)
- **No API key required** (works without token, but token gives higher rate limits)
- **Embeddings**: FREE via Hugging Face Inference API
- **LLM**: FREE via Hugging Face Inference API (Mistral, Llama, etc.)
- **Rate Limits**: 
  - Without token: ~30 requests/minute
  - With free token: Much higher limits
- **Get free token**: https://huggingface.co/settings/tokens

### Option 2: Google Gemini (FREE Tier)
- **Free tier**: 15 RPM, 1500 requests/day (more than enough for 50-100 responses)
- **Embeddings**: Use Hugging Face (FREE)
- **LLM**: Google Gemini (FREE)
- **Get API key**: https://makersuite.google.com/app/apikey

## Quick Start (Hugging Face - No API Key Required)

### Step 1: Generate Embeddings (FREE)

```bash
# No API key needed!
python embedder_hf.py
```

This creates:
- `data/faiss_index_hf.bin`
- `data/metadata_hf.jsonl`

**Note**: First request may take 30-60 seconds as Hugging Face loads the model (one-time delay).

### Step 2: Start Backend (FREE)

```bash
# Using Hugging Face for both embeddings and LLM (completely free)
python rag_backend_hf.py
```

## Using Google Gemini (Better Quality, Still FREE)

### Step 1: Get Gemini API Key
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with Google account
3. Create API key (FREE)

### Step 2: Set Environment Variable

```bash
# Windows
set GEMINI_API_KEY=your-gemini-api-key-here
set USE_GEMINI=true

# Linux/Mac
export GEMINI_API_KEY=your-gemini-api-key-here
export USE_GEMINI=true
```

### Step 3: Run Backend

```bash
python rag_backend_hf.py
```

Backend will automatically use Gemini for LLM if `USE_GEMINI=true` is set.

## Using Hugging Face Token (Optional - Higher Rate Limits)

Get a free token at: https://huggingface.co/settings/tokens

Then set:
```bash
export HUGGINGFACE_API_TOKEN=your-hf-token-here
```

This gives you higher rate limits (not required for basic usage).

## API Comparison

| Feature | Hugging Face | Google Gemini | OpenAI |
|---------|-------------|---------------|---------|
| Cost | FREE | FREE | Paid |
| Embeddings | ✅ FREE | ❌ (Use HF) | ✅ Paid |
| LLM | ✅ FREE | ✅ FREE | ✅ Paid |
| Rate Limits | 30/min (no token) | 15 RPM, 1500/day | Based on tier |
| Quality | Good | Excellent | Excellent |
| Setup | None needed | API key needed | API key needed |

## Files

- **embedder_hf.py** - Generates embeddings using Hugging Face (FREE)
- **rag_backend_hf.py** - Backend using FREE APIs (HF or Gemini)
- **embedder.py** - Original OpenAI version (paid)
- **rag_backend.py** - Original OpenAI version (paid)

## Usage Examples

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the investment objective of HDFC Flexi Cap Fund?"}'
```

## Troubleshooting

### Model Loading Delay
- First request to Hugging Face may take 30-60 seconds (model loading)
- Subsequent requests are fast
- This is normal and happens once per model

### Rate Limits
- Without HF token: ~30 requests/minute
- With HF token: Much higher limits
- Gemini: 15 RPM, 1500 requests/day (plenty for testing)

### API Errors
- **503 Error**: Model is loading, wait 30 seconds and retry
- **401 Error**: Invalid API key (check your Gemini key)
- **429 Error**: Rate limit exceeded, wait a minute

## Free Tier Limits Summary

### Hugging Face (No Token)
- ✅ Completely free
- ✅ Unlimited usage (rate limited)
- ✅ No signup required
- ~30 requests/minute

### Hugging Face (With Free Token)
- ✅ Still completely free
- ✅ Higher rate limits
- ⚠️ Requires free account signup

### Google Gemini
- ✅ 15 requests per minute
- ✅ 1,500 requests per day
- ✅ More than enough for 50-100 test responses
- ⚠️ Requires API key (free to get)

## Recommendation

**For testing (50-100 responses):**
- Use **Hugging Face** (no setup, works immediately)
- Or use **Google Gemini** for better quality (requires free API key)

Both options are completely FREE and sufficient for your needs!


