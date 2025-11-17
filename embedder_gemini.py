"""
Embedding generator using Google Gemini API
Generates embeddings for chunks and stores them in ChromaDB
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
OUTPUT_DIR = "data/chroma_db"
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
        result = genai.embed_content(
            model=model,
            content=texts,
            task_type="retrieval_document"
        )
        return result['embedding']
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


def store_in_chromadb(embeddings: List[List[float]], metadata: List[Dict], chunks: List[Dict]):
    """Store embeddings in ChromaDB."""
    try:
        import chromadb
        from chromadb.config import Settings
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(
            path=OUTPUT_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Delete existing collection if it exists
        try:
            client.delete_collection(name="mutual_fund_faq")
            logger.info("Deleted existing collection")
        except:
            pass
        
        # Create new collection
        collection = client.create_collection(
            name="mutual_fund_faq",
            metadata={"description": "Mutual Fund FAQ embeddings"}
        )
        
        # Prepare data for insertion
        ids = [chunk['chunk_id'] for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        
        # Add to collection
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata
        )
        
        logger.info(f"Stored {len(embeddings)} embeddings in ChromaDB at {OUTPUT_DIR}")
        logger.info(f"Collection: mutual_fund_faq")
        
        # Verify storage
        count = collection.count()
        logger.info(f"Verified: {count} vectors in database")
        
        return collection
        
    except ImportError:
        logger.error("ChromaDB not installed. Install with: pip install chromadb")
        raise
    except Exception as e:
        logger.error(f"Error storing in ChromaDB: {e}")
        raise


def main():
    """Main function to generate embeddings and store in vector database."""
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
        
        # Step 3: Store in ChromaDB
        logger.info(f"\nStep 3: Storing embeddings in ChromaDB...")
        collection = store_in_chromadb(embeddings, metadata, chunks)
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("EMBEDDING GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total chunks processed: {len(chunks)}")
        logger.info(f"Total embeddings created: {len(embeddings)}")
        logger.info(f"Vector database location: {OUTPUT_DIR}")
        logger.info(f"Collection name: mutual_fund_faq")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
