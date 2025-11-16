"""
FastAPI backend with RAG pipeline using FREE APIs:
- Embeddings: Hugging Face Inference API (FREE)
- LLM: Hugging Face Inference API or Google Gemini (FREE)
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Mutual Fund FAQ RAG Backend (FREE)", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = "https://api-inference.huggingface.co"
HF_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN', None)  # Optional, works without
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', None)  # Optional, for Gemini LLM
USE_GEMINI = os.getenv('USE_GEMINI', 'false').lower() == 'true'  # Set to use Gemini

# LLM configuration
LLM_PROVIDER = "gemini" if USE_GEMINI and GEMINI_API_KEY else "hf"
HF_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Free HF model
GEMINI_MODEL = "gemini-pro"  # Free Gemini model

SIMILARITY_THRESHOLD = 0.75
TOP_K = 3

# Global variables
faiss_index: Optional[faiss.Index] = None
metadata: List[Dict] = []
full_chunks: Dict[str, Dict] = {}

# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="User question", min_length=1, max_length=500)


class AnswerResponse(BaseModel):
    mode: str = Field(..., description="Response mode: FACT, REFUSAL, NO_DATA, or ERROR")
    answer: str = Field(..., description="Answer text")
    citation_url: Optional[str] = Field(None, description="Source URL for citation")


# Load FAISS index
def load_faiss_index(index_path: str) -> faiss.Index:
    """Load FAISS index from disk."""
    file_path = Path(index_path)
    if not file_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    
    try:
        index = faiss.read_index(str(file_path))
        logger.info(f"Loaded FAISS index: {index.ntotal} vectors")
        return index
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise


# Load metadata
def load_metadata(metadata_path: str) -> List[Dict]:
    """Load metadata from JSONL file."""
    file_path = Path(metadata_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        metadata_list.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Loaded {len(metadata_list)} metadata entries")
        return metadata_list
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise


# Load full chunks
def load_full_chunks(chunks_path: str) -> Dict[str, Dict]:
    """Load full chunks for better context."""
    file_path = Path(chunks_path)
    if not file_path.exists():
        logger.warning(f"Full chunks file not found: {chunks_path}")
        return {}
    
    chunks_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        chunk = json.loads(line)
                        chunk_id = chunk.get('chunk_id')
                        if chunk_id:
                            chunks_dict[chunk_id] = chunk
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Loaded {len(chunks_dict)} full chunks")
        return chunks_dict
    except Exception as e:
        logger.warning(f"Failed to load full chunks: {e}")
        return {}


# Create embedding using Hugging Face
def create_embedding_hf(text: str) -> np.ndarray:
    """Create embedding using Hugging Face API (FREE)."""
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    api_url = f"{HF_API_URL}/pipeline/feature-extraction/{EMBEDDING_MODEL}"
    payload = {"inputs": text}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 503:
            # Model loading, wait
            import time
            time.sleep(30)
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"HF API error {response.status_code}: {response.text}")
        
        embedding = response.json()
        if isinstance(embedding, list) and len(embedding) > 0:
            embedding = embedding[0]
        
        embedding_array = np.array([embedding], dtype='float32')
        faiss.normalize_L2(embedding_array)
        return embedding_array
        
    except Exception as e:
        logger.error(f"Failed to create embedding: {e}")
        raise


# Search FAISS index
def search_faiss(index: faiss.Index, query_embedding: np.ndarray, top_k: int = TOP_K):
    """Search FAISS index."""
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]


# Build prompt
def build_prompt(question: str, chunks: List[Dict]) -> str:
    """Build prompt with retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        scheme_name = chunk.get('scheme_name', 'Unknown Scheme')
        section = chunk.get('section', 'general')
        text = chunk.get('text', '')
        context_parts.append(f"Context {i} (from {scheme_name} - {section}):\n{text}")
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a factual assistant for mutual fund information. Answer the user's question using ONLY the provided context.

CRITICAL RULES:
1. Use ONLY information from the provided context. Do not use external knowledge.
2. Keep your answer to 3 sentences maximum.
3. Be factual and concise.
4. ABSOLUTELY FORBIDDEN:
   - Buy/sell recommendations
   - Portfolio advice or suggestions
   - Predictions about future performance
   - Investment advice of any kind
5. If the context doesn't contain enough information, say "I don't have enough information to answer this question based on the available data."
6. Do NOT include URLs or citations in your answer.

Context:
{context}

Question: {question}

Answer (factual, ≤3 sentences, no advice, no predictions, context only):"""

    return prompt


# Check for forbidden content
def check_for_forbidden_content(question: str) -> bool:
    """Check if question asks for forbidden content."""
    forbidden_patterns = [
        r'\b(should|should i|would you recommend|do you recommend|can you recommend|best|worst|buy|sell)\b',
        r'\b(portfolio|investment advice|should i invest|invest in)\b',
        r'\b(predict|forecast|future|will|going to|expect)\b',
        r'\b(what to buy|what to sell|which fund to choose|which is better)\b',
    ]
    
    question_lower = question.lower()
    for pattern in forbidden_patterns:
        if re.search(pattern, question_lower, re.IGNORECASE):
            return True
    return False


# Generate answer using Hugging Face
def generate_answer_hf(prompt: str) -> str:
    """Generate answer using Hugging Face Inference API (FREE)."""
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    api_url = f"{HF_API_URL}/pipeline/text-generation/{HF_LLM_MODEL}"
    
    # Format prompt for Mistral
    formatted_prompt = f"<s>[INST] {prompt} [/INST]"
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.1,
            "return_full_text": False
        }
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 503:
            import time
            time.sleep(30)
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"HF API error {response.status_code}: {response.text}")
        
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            generated_text = result[0].get('generated_text', '')
            # Clean up the response
            answer = generated_text.strip()
            # Remove any instruction tokens that might appear
            answer = re.sub(r'\[/INST\]', '', answer).strip()
            return answer
        else:
            raise Exception("Unexpected response format")
            
    except Exception as e:
        logger.error(f"Failed to generate answer via HF: {e}")
        raise


# Generate answer using Google Gemini
def generate_answer_gemini(prompt: str) -> str:
    """Generate answer using Google Gemini API (FREE)."""
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    
    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 200,
        }
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"Gemini API error {response.status_code}: {response.text}")
        
        result = response.json()
        
        if 'candidates' in result and len(result['candidates']) > 0:
            candidate = result['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                parts = candidate['content']['parts']
                if len(parts) > 0 and 'text' in parts[0]:
                    return parts[0]['text'].strip()
        
        raise Exception("Unexpected response format from Gemini")
        
    except Exception as e:
        logger.error(f"Failed to generate answer via Gemini: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load FAISS index and metadata on startup."""
    global faiss_index, metadata, full_chunks
    
    # Try Hugging Face index first, fall back to OpenAI index
    index_paths = ["data/faiss_index_hf.bin", "data/faiss_index.bin"]
    metadata_paths = ["data/metadata_hf.jsonl", "data/metadata.jsonl"]
    chunks_path = "data/processed_chunks.jsonl"
    
    index_path = None
    metadata_path = None
    
    for path in index_paths:
        if Path(path).exists():
            index_path = path
            break
    
    for path in metadata_paths:
        if Path(path).exists():
            metadata_path = path
            break
    
    if not index_path or not metadata_path:
        raise FileNotFoundError("FAISS index or metadata not found. Please run embedder.py or embedder_hf.py first.")
    
    try:
        logger.info("Loading FAISS index and metadata...")
        faiss_index = load_faiss_index(index_path)
        metadata = load_metadata(metadata_path)
        full_chunks = load_full_chunks(chunks_path)
        
        logger.info(f"Using LLM provider: {LLM_PROVIDER}")
        if LLM_PROVIDER == "gemini":
            logger.info("Using Google Gemini (FREE) for LLM")
        else:
            logger.info(f"Using Hugging Face {HF_LLM_MODEL} (FREE) for LLM")
        
        logger.info("RAG backend initialized successfully (FREE APIs)")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG backend: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "llm_provider": LLM_PROVIDER, "embedding_model": EMBEDDING_MODEL}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Answer question using RAG pipeline with FREE APIs."""
    global faiss_index, metadata, full_chunks
    
    try:
        if faiss_index is None or not metadata:
            return AnswerResponse(
                mode="ERROR",
                answer="Backend not properly initialized. Please check server logs.",
                citation_url=None
            )
        
        question = request.question.strip()
        
        if not question:
            return AnswerResponse(
                mode="ERROR",
                answer="Question cannot be empty.",
                citation_url=None
            )
        
        # Check for forbidden content
        if check_for_forbidden_content(question):
            return AnswerResponse(
                mode="REFUSAL",
                answer="I cannot provide buy/sell recommendations, portfolio advice, or predictions. I can only provide factual information.",
                citation_url=None
            )
        
        # Create embedding
        try:
            query_embedding = create_embedding_hf(question)
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return AnswerResponse(
                mode="ERROR",
                answer="Failed to process question. Please try again.",
                citation_url=None
            )
        
        # Search FAISS
        try:
            distances, indices = search_faiss(faiss_index, query_embedding, top_k=TOP_K)
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return AnswerResponse(
                mode="ERROR",
                answer="Failed to search knowledge base. Please try again.",
                citation_url=None
            )
        
        # Check similarity threshold
        if len(distances) == 0 or distances[0] < SIMILARITY_THRESHOLD:
            return AnswerResponse(
                mode="NO_DATA",
                answer="I don't have enough relevant information to answer this question based on the available data.",
                citation_url=None
            )
        
        # Get top chunks
        top_chunks = []
        citation_url = None
        
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= len(metadata):
                continue
            
            chunk_metadata = metadata[idx]
            if citation_url is None:
                citation_url = chunk_metadata.get('source_url')
            
            chunk_id = chunk_metadata.get('chunk_id')
            if chunk_id and chunk_id in full_chunks:
                text = full_chunks[chunk_id].get('text', chunk_metadata.get('text', ''))
            else:
                text = chunk_metadata.get('text', '')
            
            top_chunks.append({
                **chunk_metadata,
                'text': text
            })
        
        if not top_chunks:
            return AnswerResponse(
                mode="NO_DATA",
                answer="I don't have enough relevant information to answer this question.",
                citation_url=None
            )
        
        # Build prompt and generate answer
        try:
            prompt = build_prompt(question, top_chunks)
            
            if LLM_PROVIDER == "gemini":
                answer = generate_answer_gemini(prompt)
            else:
                answer = generate_answer_hf(prompt)
            
            if not answer or not answer.strip():
                return AnswerResponse(
                    mode="ERROR",
                    answer="Generated answer is empty. Please try again.",
                    citation_url=citation_url
                )
            
            return AnswerResponse(
                mode="FACT",
                answer=answer,
                citation_url=citation_url
            )
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return AnswerResponse(
                mode="ERROR",
                answer="Failed to generate answer. Please try again.",
                citation_url=citation_url
            )
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return AnswerResponse(
            mode="ERROR",
            answer="An unexpected error occurred. Please try again later.",
            citation_url=None
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


