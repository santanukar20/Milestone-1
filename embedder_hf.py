"""
Embedding generator using Hugging Face Inference API (FREE).
Alternative to OpenAI embeddings for cost-free usage.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Hugging Face model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Free, 384 dimensions
# Alternative options:
# "sentence-transformers/all-mpnet-base-v2"  # Better quality, 768 dimensions
# "sentence-transformers/all-MiniLM-L12-v2"  # 384 dimensions, better than L6
HF_API_URL = "https://router.huggingface.co/hf-inference/pipeline/feature-extraction"
BATCH_SIZE = 32  # HF API can handle batches

# Embedding dimension (varies by model)
EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


def get_embedding_dimension(model: str) -> int:
    """Get embedding dimension for a model."""
    return EMBEDDING_DIMENSIONS.get(model, 384)


def load_chunks(chunks_path: str) -> List[Dict]:
    """Load chunks from JSONL file."""
    file_path = Path(chunks_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")
    
    chunks = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    chunk = json.loads(line)
                    if 'text' not in chunk or not chunk.get('text'):
                        logger.warning(f"Skipping chunk on line {line_num}: missing or empty 'text' field")
                        continue
                    if 'chunk_id' not in chunk:
                        logger.warning(f"Skipping chunk on line {line_num}: missing 'chunk_id' field")
                        continue
                    chunks.append(chunk)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON on line {line_num}: {e}")
                    continue
        
        if not chunks:
            raise ValueError(f"No valid chunks found in {chunks_path}")
        
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
        return chunks
        
    except Exception as e:
        logger.error(f"Error loading chunks from {chunks_path}: {e}")
        raise


def create_embeddings_hf(texts: List[str], model: str = EMBEDDING_MODEL, api_token: Optional[str] = None) -> List[List[float]]:
    """
    Create embeddings using Hugging Face Inference API (FREE).
    
    Args:
        texts: List of texts to embed
        model: Hugging Face model name
        api_token: Optional HF API token (for higher rate limits)
        
    Returns:
        List of embedding vectors
        
    Raises:
        Exception: If embedding creation fails
    """
    if not texts:
        return []
    
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    
    # Prepare request
    api_url = f"https://router.huggingface.co/hf-inference/pipeline/feature-extraction/{model}"
    
    # HF API can handle single or multiple inputs
    payload = {"inputs": texts if len(texts) > 1 else texts[0]}
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 503:
            # Model is loading, wait and retry
            logger.info("Model is loading, waiting 30 seconds...")
            time.sleep(30)
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            error_msg = response.text
            raise Exception(f"HF API error {response.status_code}: {error_msg}")
        
        embeddings = response.json()
        
        # Handle both single and batch responses
        if isinstance(embeddings, list) and len(embeddings) > 0:
            if isinstance(embeddings[0], list):
                # Batch response: list of lists
                return embeddings
            else:
                # Single response: wrap in list
                return [embeddings]
        else:
            raise Exception("Unexpected response format from HF API")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise Exception(f"Failed to create embeddings via HF API: {e}")
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise


def embed_chunks(chunks: List[Dict], model: str = EMBEDDING_MODEL, batch_size: int = BATCH_SIZE) -> Tuple[np.ndarray, List[Dict]]:
    """
    Embed all chunks in batches using Hugging Face API.
    
    Args:
        chunks: List of chunk dictionaries
        model: Embedding model name
        batch_size: Number of chunks to embed per batch
        
    Returns:
        Tuple of (embeddings array, metadata list with vector_ids)
    """
    all_embeddings = []
    metadata = []
    
    # Get API token if available (optional, works without but has lower rate limits)
    api_token = os.getenv('HUGGINGFACE_API_TOKEN', None)
    if api_token:
        logger.info("Using Hugging Face API token for higher rate limits")
    else:
        logger.info("Using Hugging Face API without token (free tier, lower rate limits)")
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    expected_dimension = get_embedding_dimension(model)
    
    logger.info(f"Creating embeddings for {len(chunks)} chunks using model: {model}")
    logger.info(f"Expected dimension: {expected_dimension}")
    logger.info(f"Processing in {total_batches} batch(es)...")
    
    for batch_idx in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in batch_chunks]
        
        try:
            # Create embeddings for this batch
            batch_embeddings = create_embeddings_hf(texts, model, api_token)
            
            if len(batch_embeddings) != len(batch_chunks):
                raise ValueError(
                    f"Embedding count mismatch: expected {len(batch_chunks)}, "
                    f"got {len(batch_embeddings)}"
                )
            
            # Add embeddings and metadata
            for chunk, embedding in zip(batch_chunks, batch_embeddings):
                vector_id = len(all_embeddings)
                
                # Validate embedding dimension
                if len(embedding) != expected_dimension:
                    logger.warning(
                        f"Embedding dimension mismatch for chunk {chunk.get('chunk_id', 'unknown')}: "
                        f"expected {expected_dimension}, got {len(embedding)}"
                    )
                
                all_embeddings.append(embedding)
                
                # Create metadata entry
                metadata_entry = {
                    'vector_id': vector_id,
                    'chunk_id': chunk.get('chunk_id'),
                    'scheme_name': chunk.get('scheme_name'),
                    'section': chunk.get('section'),
                    'text': chunk.get('text', '')[:200],
                    'source_url': chunk.get('source_url'),
                    'last_scraped_at': chunk.get('last_scraped_at')
                }
                metadata.append(metadata_entry)
            
            logger.info(f"Batch {batch_num}/{total_batches} completed successfully")
            
            # Rate limiting for free tier (no token)
            if not api_token and batch_num < total_batches:
                time.sleep(1)  # Small delay between batches
            
        except Exception as e:
            logger.error(f"Failed to process batch {batch_num}/{total_batches}: {e}")
            raise
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings, dtype='float32')
    
    logger.info(f"Successfully created embeddings for {len(all_embeddings)} chunks")
    logger.info(f"Embedding shape: {embeddings_array.shape}")
    
    return embeddings_array, metadata


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Build FAISS index with Inner Product similarity."""
    dimension = embeddings.shape[1]
    n_vectors = embeddings.shape[0]
    
    logger.info(f"Building FAISS index: {n_vectors} vectors, dimension {dimension}")
    
    # Make a copy and normalize for cosine similarity
    embeddings_normalized = embeddings.copy().astype('float32')
    faiss.normalize_L2(embeddings_normalized)
    
    # Create Flat Index with Inner Product
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_normalized)
    
    logger.info(f"FAISS index built successfully: {index.ntotal} vectors indexed")
    return index


def save_index(index: faiss.Index, index_path: str) -> None:
    """Save FAISS index to disk."""
    file_path = Path(index_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        faiss.write_index(index, str(file_path))
        logger.info(f"FAISS index saved to {index_path}")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")
        raise


def save_metadata(metadata: List[Dict], metadata_path: str) -> None:
    """Save metadata to JSONL file."""
    file_path = Path(metadata_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in metadata:
                json_line = json.dumps(entry, ensure_ascii=False)
                f.write(json_line + '\n')
        
        logger.info(f"Metadata saved to {metadata_path} ({len(metadata)} entries)")
    except Exception as e:
        logger.error(f"Failed to save metadata: {e}")
        raise


def main() -> None:
    """Main function to generate embeddings and build FAISS index."""
    input_file = "data/processed_chunks.jsonl"
    index_file = "data/faiss_index_hf.bin"  # Different filename to avoid conflicts
    metadata_file = "data/metadata_hf.jsonl"
    
    logger.info("=" * 60)
    logger.info("EMBEDDING GENERATOR - Hugging Face (FREE)")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load chunks
        logger.info(f"\nStep 1: Loading chunks from {input_file}...")
        chunks = load_chunks(input_file)
        logger.info(f"Loaded {len(chunks)} chunks")
        
        # Step 2: Create embeddings (FREE via HF API)
        logger.info("\nStep 2: Creating embeddings using Hugging Face API (FREE)...")
        embeddings, metadata = embed_chunks(chunks, model=EMBEDDING_MODEL, batch_size=BATCH_SIZE)
        logger.info(f"Created {len(embeddings)} embeddings")
        
        # Step 3: Build FAISS index
        logger.info("\nStep 3: Building FAISS index...")
        index = build_faiss_index(embeddings)
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        
        # Step 4: Save index and metadata
        logger.info("\nStep 4: Saving index and metadata...")
        save_index(index, index_file)
        save_metadata(metadata, metadata_file)
        
        # Print summary
        actual_dimension = embeddings.shape[1] if len(embeddings) > 0 else 0
        logger.info("\n" + "=" * 60)
        logger.info("EMBEDDING SUMMARY (Hugging Face)")
        logger.info("=" * 60)
        logger.info(f"Total chunks processed:    {len(chunks)}")
        logger.info(f"Total vectors created:     {len(embeddings)}")
        logger.info(f"Embedding dimension:       {actual_dimension}")
        logger.info(f"Model used:                {EMBEDDING_MODEL}")
        logger.info(f"FAISS index vectors:       {index.ntotal}")
        logger.info(f"Index file:                {index_file}")
        logger.info(f"Metadata file:             {metadata_file}")
        logger.info("=" * 60)
        logger.info("\nEmbedding generation completed successfully!")
        logger.info("Note: Hugging Face Inference API is FREE (no API key required)")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure you've run chunker.py first")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


