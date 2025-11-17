"""
Embedding generator using Google Gemini API
Generates embeddings for chunks and stores them in JSON format
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = "AIzaSyDrgz_VSsVQAGlYcL1vhwpF4FKOAUxstuM"
EMBEDDING_MODEL = "models/text-embedding-004"  # Latest Gemini embedding model
INPUT_FILE = "data/processed_chunks.jsonl"
OUTPUT_FILE = "data/embeddings.json"
BATCH_SIZE = 100  # Gemini can handle large batches

def load_chunks(file_path: str) -> List[Dict]:
    """Load chunks from JSONL file."""
    chunks = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        logger.info(f"Loaded {len(chunks)} chunks from {file_path}")
        return chunks
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading chunks: {e}")
        raise


def create_embeddings_batch(texts: List[str], model: str) -> List[List[float]]:
    """Create embeddings using Gemini API."""
    try:
        # For multiple texts, we need to call the API for each text
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model=model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        return embeddings
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise


def embed_chunks(chunks: List[Dict], model: str = EMBEDDING_MODEL, batch_size: int = BATCH_SIZE) -> Tuple[List[List[float]], List[Dict]]:
    """Generate embeddings for all chunks."""
    logger.info(f"Creating embeddings for {len(chunks)} chunks using model: {model}")
    
    texts = [chunk['text'] for chunk in chunks]
    metadata = [
        {
            'chunk_id': chunk['chunk_id'],
            'scheme_name': chunk['scheme_name'],
            'section': chunk.get('section', ''),
            'source_url': chunk['source_url'],
            'last_scraped_at': chunk.get('last_scraped_at', '')
        }
        for chunk in chunks
    ]
    
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    logger.info(f"Processing in {total_batches} batch(es)...")
    
    for i in range(0, len(texts), batch_size):
        batch_num = (i // batch_size) + 1
        batch_texts = texts[i:i + batch_size]
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} chunks)...")
        
        try:
            batch_embeddings = create_embeddings_batch(batch_texts, model)
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Batch {batch_num}/{total_batches} completed")
            
            # Rate limiting
            if batch_num < total_batches:
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed to process batch {batch_num}/{total_batches}: {e}")
            raise
    
    logger.info(f"Generated {len(all_embeddings)} embeddings")
    return all_embeddings, metadata


def save_embeddings(embeddings: List[List[float]], metadata: List[Dict], chunks: List[Dict], output_file: str):
    """Save embeddings and metadata to JSON file."""
    try:
        # Prepare data for saving
        data = {
            'embeddings': embeddings,
            'metadata': metadata,
            'documents': [chunk['text'] for chunk in chunks]
        }
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(embeddings)} embeddings to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving embeddings: {e}")
        raise


def main():
    """Main function to generate embeddings and save to file."""
    try:
        logger.info("=" * 60)
        logger.info("EMBEDDING GENERATOR - Google Gemini")
        logger.info("=" * 60)
        
        # Configure Gemini API
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured")
        
        # Step 1: Load chunks
        logger.info(f"\nStep 1: Loading chunks from {INPUT_FILE}...")
        chunks = load_chunks(INPUT_FILE)
        logger.info(f"Loaded {len(chunks)} chunks")
        
        # Step 2: Create embeddings
        logger.info(f"\nStep 2: Creating embeddings using Gemini API...")
        embeddings, metadata = embed_chunks(chunks, model=EMBEDDING_MODEL, batch_size=BATCH_SIZE)
        
        # Step 3: Save embeddings
        logger.info(f"\nStep 3: Saving embeddings to {OUTPUT_FILE}...")
        save_embeddings(embeddings, metadata, chunks, OUTPUT_FILE)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total chunks processed: {len(chunks)}")
        logger.info(f"Total embeddings created: {len(embeddings)}")
        logger.info(f"Embeddings saved to: {OUTPUT_FILE}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
