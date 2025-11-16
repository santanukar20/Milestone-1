"""
Embedding generator for Mutual Fund FAQ RAG chatbot.
Converts processed chunks into vectors and stores them in FAISS.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed. Install with: pip install openai")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-3-large"
EXPECTED_DIMENSION = 3072  # Expected dimension for text-embedding-3-large

# Batch size for API calls (OpenAI allows batches up to 2048, but we'll use smaller batches)
BATCH_SIZE = 100


def load_chunks(chunks_path: str) -> List[Dict]:
    """
    Load chunks from JSONL file.
    
    Args:
        chunks_path: Path to processed_chunks.jsonl
        
    Returns:
        List of chunk dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
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
                    
                    # Validate required fields
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


def initialize_openai_client() -> OpenAI:
    """
    Initialize OpenAI client with API key validation.
    
    Returns:
        OpenAI client instance
        
    Raises:
        ValueError: If API key is not set
        Exception: If client initialization fails
    """
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with: export OPENAI_API_KEY='your-key-here'"
        )
    
    try:
        client = OpenAI(api_key=api_key)
        # Test the client with a simple call (this will fail fast if key is invalid)
        logger.info("Validating OpenAI API key...")
        # We'll validate by trying to create an embedding
        return client
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise


def create_embeddings(
    client: OpenAI,
    texts: List[str],
    model: str = EMBEDDING_MODEL
) -> List[List[float]]:
    """
    Create embeddings for a batch of texts.
    
    Args:
        client: OpenAI client instance
        texts: List of texts to embed
        model: Embedding model to use
        
    Returns:
        List of embedding vectors
        
    Raises:
        Exception: If embedding creation fails
    """
    if not texts:
        return []
    
    try:
        response = client.embeddings.create(
            model=model,
            input=texts
        )
        
        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]
        
        return embeddings
        
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise


def embed_chunks(
    client: OpenAI,
    chunks: List[Dict],
    batch_size: int = BATCH_SIZE
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Embed all chunks in batches.
    
    Args:
        client: OpenAI client instance
        chunks: List of chunk dictionaries
        batch_size: Number of chunks to embed per batch
        
    Returns:
        Tuple of (embeddings array, metadata list with vector_ids)
    """
    all_embeddings = []
    metadata = []
    
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    logger.info(f"Creating embeddings for {len(chunks)} chunks in {total_batches} batch(es)...")
    
    for batch_idx in range(0, len(chunks), batch_size):
        batch_chunks = chunks[batch_idx:batch_idx + batch_size]
        batch_num = (batch_idx // batch_size) + 1
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
        
        # Extract texts for embedding
        texts = [chunk['text'] for chunk in batch_chunks]
        
        try:
            # Create embeddings for this batch
            batch_embeddings = create_embeddings(client, texts, EMBEDDING_MODEL)
            
            if len(batch_embeddings) != len(batch_chunks):
                raise ValueError(
                    f"Embedding count mismatch: expected {len(batch_chunks)}, "
                    f"got {len(batch_embeddings)}"
                )
            
            # Detect dimension from first embedding (fail fast if inconsistent)
            if batch_embeddings:
                detected_dimension = len(batch_embeddings[0])
                if detected_dimension != EXPECTED_DIMENSION:
                    logger.warning(
                        f"Detected dimension {detected_dimension} differs from expected "
                        f"{EXPECTED_DIMENSION}. Using detected dimension."
                    )
            
            # Add embeddings and metadata
            for i, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                vector_id = len(all_embeddings)
                
                # Validate embedding dimension consistency
                if i > 0 and len(embedding) != len(batch_embeddings[0]):
                    raise ValueError(
                        f"Embedding dimension inconsistency: first embedding has "
                        f"{len(batch_embeddings[0])} dimensions, but chunk {chunk.get('chunk_id', 'unknown')} "
                        f"has {len(embedding)} dimensions"
                    )
                
                all_embeddings.append(embedding)
                
                # Create metadata entry with vector_id
                metadata_entry = {
                    'vector_id': vector_id,
                    'chunk_id': chunk.get('chunk_id'),
                    'scheme_name': chunk.get('scheme_name'),
                    'section': chunk.get('section'),
                    'text': chunk.get('text', '')[:200],  # Store first 200 chars for reference
                    'source_url': chunk.get('source_url'),
                    'last_scraped_at': chunk.get('last_scraped_at')
                }
                metadata.append(metadata_entry)
            
            logger.info(f"Batch {batch_num}/{total_batches} completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to process batch {batch_num}/{total_batches}: {e}")
            raise
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings, dtype='float32')
    
    logger.info(f"Successfully created embeddings for {len(all_embeddings)} chunks")
    logger.info(f"Embedding shape: {embeddings_array.shape}")
    
    return embeddings_array, metadata


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build FAISS index with Inner Product similarity.
    
    Args:
        embeddings: Numpy array of embeddings (shape: [n_vectors, dimension])
        
    Returns:
        FAISS index
    """
    dimension = embeddings.shape[1]
    n_vectors = embeddings.shape[0]
    
    logger.info(f"Building FAISS index: {n_vectors} vectors, dimension {dimension}")
    
    # Make a copy of embeddings to avoid modifying the original
    embeddings_normalized = embeddings.copy().astype('float32')
    
    # Normalize embeddings for cosine similarity (using Inner Product on normalized vectors)
    # FAISS Inner Product on normalized vectors = cosine similarity
    faiss.normalize_L2(embeddings_normalized)
    
    # Create Flat Index with Inner Product
    # FlatIP uses Inner Product metric (perfect for normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    
    # Add vectors to index
    index.add(embeddings_normalized)
    
    logger.info(f"FAISS index built successfully: {index.ntotal} vectors indexed")
    
    return index


def save_index(index: faiss.Index, index_path: str) -> None:
    """
    Save FAISS index to disk.
    
    Args:
        index: FAISS index to save
        index_path: Path to save index
    """
    file_path = Path(index_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        faiss.write_index(index, str(file_path))
        logger.info(f"FAISS index saved to {index_path}")
    except Exception as e:
        logger.error(f"Failed to save FAISS index: {e}")
        raise


def save_metadata(metadata: List[Dict], metadata_path: str) -> None:
    """
    Save metadata to JSONL file.
    
    Args:
        metadata: List of metadata dictionaries
        metadata_path: Path to save metadata
    """
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
    index_file = "data/faiss_index.bin"
    metadata_file = "data/metadata.jsonl"
    
    logger.info("=" * 60)
    logger.info("EMBEDDING GENERATOR - Building FAISS Index")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load chunks
        logger.info(f"\nStep 1: Loading chunks from {input_file}...")
        chunks = load_chunks(input_file)
        logger.info(f"Loaded {len(chunks)} chunks")
        
        # Step 2: Initialize OpenAI client
        logger.info("\nStep 2: Initializing OpenAI client...")
        client = initialize_openai_client()
        logger.info("OpenAI client initialized successfully")
        
        # Step 3: Create embeddings
        logger.info("\nStep 3: Creating embeddings...")
        embeddings, metadata = embed_chunks(client, chunks, batch_size=BATCH_SIZE)
        logger.info(f"Created {len(embeddings)} embeddings")
        
        # Step 4: Build FAISS index
        logger.info("\nStep 4: Building FAISS index...")
        index = build_faiss_index(embeddings)
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        
        # Step 5: Save index and metadata
        logger.info("\nStep 5: Saving index and metadata...")
        save_index(index, index_file)
        save_metadata(metadata, metadata_file)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("EMBEDDING SUMMARY")
        logger.info("=" * 60)
        # Get actual dimension from embeddings
        actual_dimension = embeddings.shape[1] if len(embeddings) > 0 else 0
        
        logger.info(f"Total chunks processed:    {len(chunks)}")
        logger.info(f"Total vectors created:     {len(embeddings)}")
        logger.info(f"Embedding dimension:       {actual_dimension}")
        logger.info(f"FAISS index vectors:       {index.ntotal}")
        logger.info(f"Index file:                {index_file}")
        logger.info(f"Metadata file:             {metadata_file}")
        logger.info("=" * 60)
        
        logger.info("\nEmbedding generation completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure you've run chunker.py first to generate processed_chunks.jsonl")
        sys.exit(1)
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

