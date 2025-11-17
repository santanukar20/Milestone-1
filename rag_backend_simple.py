"""
Lightweight RAG backend using pre-computed Gemini embeddings
Answers questions using mutual fund FAQ data
"""

import json
import logging
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
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
GEMBEDDING_MODEL = "models/text-embedding-004"
GENERATIVE_MODEL = "gemini-2.0-flash"
EMBEDDINGS_FILE = "data/embeddings.json"

class MutualFundRAG:
    """RAG system for answering mutual fund questions."""
    
    def __init__(self, embeddings_file: str = EMBEDDINGS_FILE):
        """Initialize RAG system with pre-computed embeddings."""
        self.embeddings_file = embeddings_file
        self.embeddings = []
        self.metadata = []
        self.documents = []
        self.load_embeddings()
        
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel(GENERATIVE_MODEL)
        
    def load_embeddings(self):
        """Load pre-computed embeddings from file."""
        try:
            with open(self.embeddings_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self.embeddings = np.array(data['embeddings'])
            self.metadata = data['metadata']
            self.documents = data['documents']
            
            logger.info(f"Loaded {len(self.embeddings)} embeddings from {self.embeddings_file}")
            logger.info(f"Embedding dimension: {len(self.embeddings[0])}")
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise
    
    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a_arr = np.array(a)
        b_arr = np.array(b)
        return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    
    def find_relevant_chunks(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Find most relevant chunks for a query using cosine similarity."""
        try:
            # Generate embedding for query
            result = genai.embed_content(
                model=GEMBEDDING_MODEL,
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = np.array(result['embedding'])
            
            # Calculate similarities with all chunks
            similarities = []
            for i, chunk_embedding in enumerate(self.embeddings):
                similarity = self.cosine_similarity(query_embedding.tolist(), chunk_embedding.tolist())
                similarities.append((i, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding relevant chunks: {e}")
            return []
    
    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """Generate answer using Gemini with provided context."""
        try:
            # Prepare context
            context = "\n\n".join(context_chunks)
            
            # Create prompt
            prompt = f"""
You are a helpful assistant answering questions about mutual funds. Use the following context to answer the question accurately. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {query}

Answer:"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Sorry, I encountered an error while generating the answer."
    
    def ask(self, question: str, top_k: int = 3) -> str:
        """Answer a question using RAG approach."""
        logger.info(f"Processing question: {question}")
        
        # Find relevant chunks
        relevant_indices = self.find_relevant_chunks(question, top_k)
        
        if not relevant_indices:
            return "Sorry, I couldn't find relevant information to answer your question."
        
        # Get context chunks
        context_chunks = [self.documents[idx] for idx, _ in relevant_indices]
        
        # Log relevant chunks
        logger.info(f"Found {len(context_chunks)} relevant chunks")
        for i, (idx, similarity) in enumerate(relevant_indices):
            meta = self.metadata[idx]
            logger.info(f"  {i+1}. {meta['scheme_name']} (similarity: {similarity:.3f})")
        
        # Generate answer
        answer = self.generate_answer(question, context_chunks)
        return answer

def main():
    """Main function to demonstrate the RAG system."""
    try:
        logger.info("=" * 60)
        logger.info("MUTUAL FUND FAQ CHATBOT - RAG SYSTEM")
        logger.info("=" * 60)
        
        # Initialize RAG system
        rag = MutualFundRAG()
        
        # Example questions
        questions = [
            "What is the expense ratio of HDFC Flexi Cap Fund?",
            "What is the minimum SIP amount for HDFC Mid Cap Fund?",
            "What is the exit load for HDFC Small Cap Fund?",
            "Who manages the HDFC Large Cap Fund?",
            "What are the returns since inception for HDFC Balanced Advantage Fund?"
        ]
        
        # Answer questions
        for i, question in enumerate(questions, 1):
            logger.info(f"\nQuestion {i}: {question}")
            answer = rag.ask(question)
            logger.info(f"Answer: {answer}")
            logger.info("-" * 60)
        
        # Interactive mode
        logger.info("\n" + "=" * 60)
        logger.info("INTERACTIVE MODE - Ask your own questions!")
        logger.info("Type 'quit' to exit")
        logger.info("=" * 60)
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if not question:
                    continue
                    
                answer = rag.ask(question)
                print(f"\nAnswer: {answer}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                print("Sorry, I encountered an error. Please try again.")
        
        logger.info("Thank you for using the Mutual Fund FAQ Chatbot!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
