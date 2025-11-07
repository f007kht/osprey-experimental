"""Embedding generation utilities.

Module 1 - Osprey Backend: Document processing application.
"""

import streamlit as st
from typing import List, Optional, Dict, Any

# Default embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Check availability
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False


@st.cache_resource
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """
    Get or create a cached SentenceTransformer model instance.
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        SentenceTransformer instance
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
    
    return SentenceTransformer(model_name)


def generate_embeddings(
    chunk_texts: List[str],
    embedding_config: Optional[Dict[str, Any]] = None
) -> List[List[float]]:
    """
    Generate embeddings for text chunks.
    
    Args:
        chunk_texts: List of text chunks to embed
        embedding_config: Configuration dictionary with:
            - use_remote: If True, use VoyageAI; if False, use local SentenceTransformer
            - model_name: Model name for local embeddings
            - api_key: VoyageAI API key (required if use_remote=True)
    
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    if not embedding_config:
        embedding_config = {"use_remote": False, "model_name": DEFAULT_EMBEDDING_MODEL}
    
    if embedding_config.get("use_remote", False):
        # Use VoyageAI for remote embeddings
        if not VOYAGEAI_AVAILABLE:
            raise ImportError("voyageai not available. Install with: pip install voyageai")
        
        api_key = embedding_config.get("api_key")
        if not api_key:
            raise ValueError("VoyageAI API key required when use_remote=True")
        
        vo = voyageai.Client(api_key=api_key)
        # Use contextualized_embed for better quality
        results = vo.contextualized_embed(
            inputs=[[text] for text in chunk_texts],
            model="voyage-context-3",
            input_type="document"
        )
        
        embeddings = [result.embeddings[0] for result in results.results]
    else:
        # Use local SentenceTransformer
        model_name = embedding_config.get("model_name", DEFAULT_EMBEDDING_MODEL)
        model = get_embedding_model(model_name)
        embeddings = model.encode(chunk_texts, show_progress_bar=False)
        # Convert numpy arrays to lists
        embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
    
    return embeddings

