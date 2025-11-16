"""
FastAPI backend with RAG pipeline for Mutual Fund FAQ chatbot.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Mutual Fund FAQ RAG Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG components
faiss_index: Optional[faiss.Index] = None
metadata: List[Dict] = []
full_chunks: Dict[str, Dict] = {}  # Map chunk_id to full chunk for better context
openai_client: Optional[OpenAI] = None
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "gpt-4o-mini"
SIMILARITY_THRESHOLD = 0.75  # Minimum similarity score
TOP_K = 3  # Number of chunks to retrieve


# Request/Response models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="User question", min_length=1, max_length=500)


class AnswerResponse(BaseModel):
    mode: str = Field(..., description="Response mode: FACT, REFUSAL, NO_DATA, or ERROR")
    answer: str = Field(..., description="Answer text")
    citation_url: Optional[str] = Field(None, description="Source URL for citation")


# Load FAISS index
def load_faiss_index(index_path: str) -> faiss.Index:
    """
    Load FAISS index from disk.
    
    Args:
        index_path: Path to FAISS index file
        
    Returns:
        FAISS index
        
    Raises:
        FileNotFoundError: If index file doesn't exist
        Exception: If index loading fails
    """
    file_path = Path(index_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    
    try:
        index = faiss.read_index(str(file_path))
        logger.info(f"Loaded FAISS index from {index_path}: {index.ntotal} vectors")
        return index
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise


# Load full chunks for better context
def load_full_chunks(chunks_path: str) -> Dict[str, Dict]:
    """
    Load full chunks from processed_chunks.jsonl for better context.
    
    Args:
        chunks_path: Path to processed_chunks.jsonl
        
    Returns:
        Dictionary mapping chunk_id to full chunk
        
    Raises:
        FileNotFoundError: If chunks file doesn't exist
    """
    file_path = Path(chunks_path)
    
    if not file_path.exists():
        logger.warning(f"Full chunks file not found: {chunks_path}, using metadata only")
        return {}
    
    chunks_dict = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    chunk = json.loads(line)
                    chunk_id = chunk.get('chunk_id')
                    if chunk_id:
                        chunks_dict[chunk_id] = chunk
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse chunk line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(chunks_dict)} full chunks from {chunks_path}")
        return chunks_dict
        
    except Exception as e:
        logger.warning(f"Failed to load full chunks: {e}, using metadata only")
        return {}


# Load metadata
def load_metadata(metadata_path: str) -> List[Dict]:
    """
    Load metadata from JSONL file.
    
    Args:
        metadata_path: Path to metadata JSONL file
        
    Returns:
        List of metadata dictionaries
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
        Exception: If metadata loading fails
    """
    file_path = Path(metadata_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata_list = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    metadata_list.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata line {line_num}: {e}")
                    continue
        
        logger.info(f"Loaded {len(metadata_list)} metadata entries from {metadata_path}")
        return metadata_list
        
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        raise


# Initialize OpenAI client
def initialize_openai_client() -> OpenAI:
    """
    Initialize OpenAI client.
    
    Returns:
        OpenAI client instance
        
    Raises:
        ValueError: If API key is not set
    """
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with: export OPENAI_API_KEY='your-key-here'"
        )
    
    client = OpenAI(api_key=api_key)
    logger.info("OpenAI client initialized")
    return client


# Create embedding
def create_embedding(client: OpenAI, text: str) -> np.ndarray:
    """
    Create embedding for text.
    
    Args:
        client: OpenAI client
        text: Text to embed
        
    Returns:
        Embedding vector as numpy array
        
    Raises:
        Exception: If embedding creation fails
    """
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text]
        )
        
        embedding = response.data[0].embedding
        embedding_array = np.array([embedding], dtype='float32')
        
        # Normalize for cosine similarity (same as in index)
        faiss.normalize_L2(embedding_array)
        
        return embedding_array
        
    except Exception as e:
        logger.error(f"Failed to create embedding: {e}")
        raise


# Search FAISS index
def search_faiss(index: faiss.Index, query_embedding: np.ndarray, top_k: int = TOP_K) -> tuple:
    """
    Search FAISS index for similar vectors.
    
    Args:
        index: FAISS index
        query_embedding: Query embedding vector
        top_k: Number of top results to return
        
    Returns:
        Tuple of (distances, indices)
    """
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]


# Build prompt for LLM
def build_prompt(question: str, chunks: List[Dict]) -> str:
    """
    Build prompt for LLM with retrieved chunks.
    
    Args:
        question: User question
        chunks: Retrieved chunks with metadata
        
    Returns:
        Formatted prompt string
    """
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
   - Buy/sell recommendations (e.g., "you should buy", "I recommend selling")
   - Portfolio advice or suggestions (e.g., "you should invest in", "consider adding")
   - Predictions about future performance (e.g., "will perform", "is expected to", "likely to")
   - Investment advice of any kind (e.g., "good investment", "worth investing")
5. If the context doesn't contain enough information to answer, say "I don't have enough information to answer this question based on the available data."
6. Do NOT include URLs or citations in your answer.
7. Do NOT generate a citation URL. The backend will handle citations.
8. Only state facts from the context. Do not add opinions or interpretations.

Context:
{context}

Question: {question}

Answer (factual, ≤3 sentences, no advice, no predictions, context only):"""

    return prompt


# Check if question asks for advice/recommendations
def check_for_forbidden_content(question: str) -> bool:
    """
    Check if question asks for forbidden content (advice, recommendations, predictions).
    
    Args:
        question: User question
        
    Returns:
        True if question asks for forbidden content, False otherwise
    """
    forbidden_patterns = [
        r'\b(should|should i|would you recommend|do you recommend|can you recommend|best|worst|buy|sell)\b',
        r'\b(portfolio|investment advice|should i invest|invest in)\b',
        r'\b(predict|forecast|future|will|going to|expect)\b',
        r'\b(what to buy|what to sell|which fund to choose|which is better)\b',
        r'\b(advice|suggestion|recommendation)\b'
    ]
    
    import re
    question_lower = question.lower()
    
    for pattern in forbidden_patterns:
        if re.search(pattern, question_lower, re.IGNORECASE):
            return True
    
    return False


# Generate answer using LLM
def generate_answer(client: OpenAI, prompt: str) -> str:
    """
    Generate answer using LLM.
    
    Args:
        client: OpenAI client
        prompt: Prompt for LLM
        
    Returns:
        Generated answer text
        
    Raises:
        Exception: If LLM call fails
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a factual assistant that provides information about mutual funds based on provided context. Never give investment advice, recommendations, or predictions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
            max_tokens=200  # Limit to ~3 sentences
        )
        
        answer = response.choices[0].message.content.strip()
        return answer
        
    except Exception as e:
        logger.error(f"Failed to generate answer: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """Load FAISS index and metadata on startup."""
    global faiss_index, metadata, full_chunks, openai_client
    
    index_path = "data/faiss_index.bin"
    metadata_path = "data/metadata.jsonl"
    chunks_path = "data/processed_chunks.jsonl"
    
    try:
        logger.info("Loading FAISS index and metadata...")
        faiss_index = load_faiss_index(index_path)
        metadata = load_metadata(metadata_path)
        full_chunks = load_full_chunks(chunks_path)  # Load full chunks for better context
        openai_client = initialize_openai_client()
        
        # Validate metadata length matches index
        if len(metadata) != faiss_index.ntotal:
            logger.warning(
                f"Metadata count ({len(metadata)}) doesn't match index size ({faiss_index.ntotal})"
            )
        
        logger.info("RAG backend initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG backend: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer question using RAG pipeline.
    
    Args:
        request: Question request with question text
        
    Returns:
        Answer response with mode, answer, and citation URL
    """
    global faiss_index, metadata, full_chunks, openai_client
    
    try:
        # Check if backend is initialized
        if faiss_index is None or not metadata or openai_client is None:
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
                answer="I cannot provide buy/sell recommendations, portfolio advice, or predictions about mutual funds. I can only provide factual information based on available data.",
                citation_url=None
            )
        
        # Create embedding for question
        try:
            query_embedding = create_embedding(openai_client, question)
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return AnswerResponse(
                mode="ERROR",
                answer="Failed to process question. Please try again.",
                citation_url=None
            )
        
        # Search FAISS index
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
        
        # Get top chunks from metadata
        top_chunks = []
        citation_url = None
        
        for i, idx in enumerate(indices):
            if idx < 0 or idx >= len(metadata):
                continue
            
            chunk_metadata = metadata[idx]
            similarity_score = float(distances[i])
            
            # Use first valid chunk's source_url as citation
            if citation_url is None:
                citation_url = chunk_metadata.get('source_url')
            
            top_chunks.append({
                **chunk_metadata,
                'similarity_score': similarity_score
            })
        
        if not top_chunks:
            return AnswerResponse(
                mode="NO_DATA",
                answer="I don't have enough relevant information to answer this question based on the available data.",
                citation_url=None
            )
        
        # Build prompt with retrieved chunks
        # Use full chunks if available, otherwise fall back to metadata
        chunks_for_prompt = []
        for chunk in top_chunks:
            chunk_id = chunk.get('chunk_id')
            
            # Try to get full chunk text if available
            if chunk_id and chunk_id in full_chunks:
                full_chunk = full_chunks[chunk_id]
                text = full_chunk.get('text', chunk.get('text', ''))
            else:
                text = chunk.get('text', '')  # Use metadata text (truncated)
            
            chunk_for_prompt = {
                'scheme_name': chunk.get('scheme_name'),
                'section': chunk.get('section'),
                'text': text
            }
            chunks_for_prompt.append(chunk_for_prompt)
        
        # Build prompt and generate answer
        try:
            prompt = build_prompt(question, chunks_for_prompt)
            answer = generate_answer(openai_client, prompt)
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return AnswerResponse(
                mode="ERROR",
                answer="Failed to generate answer. Please try again.",
                citation_url=citation_url  # Still return citation if we have it
            )
        
        # Validate answer is not empty
        if not answer or not answer.strip():
            return AnswerResponse(
                mode="ERROR",
                answer="Generated answer is empty. Please try again.",
                citation_url=citation_url
            )
        
        # Return factual answer
        return AnswerResponse(
            mode="FACT",
            answer=answer,
            citation_url=citation_url
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in /ask endpoint: {e}", exc_info=True)
        return AnswerResponse(
            mode="ERROR",
            answer="An unexpected error occurred. Please try again later.",
            citation_url=None
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

