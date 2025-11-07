"""MongoDB storage operations.

Module 1 - Osprey Backend: Document processing application.
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

try:
    from pymongo import MongoClient
    from pymongo.operations import SearchIndexModel
    from pymongo.errors import OperationFailure
    import certifi
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    MongoClient = None
    SearchIndexModel = None
    OperationFailure = None
    certifi = None

from app.config import get_settings, QA_SCHEMA_VERSION
from app.utils.text_processing import chunk_document
from app.utils.embeddings import generate_embeddings, DEFAULT_EMBEDDING_MODEL
from utils.mongodb_helpers import split_for_mongo, bson_len, MAX_BSON_SAFE


def sanitize_for_mongodb(obj: Any) -> Any:
    """
    Recursively sanitize data for MongoDB BSON compatibility.
    
    Converts unsupported types (like sets, complex numbers) to supported types.
    
    Args:
        obj: Object to sanitize (can be dict, list, or primitive)
        
    Returns:
        Sanitized object compatible with BSON
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_mongodb(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_mongodb(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)  # Convert sets to lists
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    elif hasattr(obj, '__dict__'):
        # Convert objects to dicts
        return sanitize_for_mongodb(obj.__dict__)
    else:
        # Fallback: convert to string
        return str(obj)


def store_in_mongodb(
    document: Any,
    original_filename: str,
    file_size: int,
    mongodb_connection_string: str,
    database_name: str,
    collection_name: str,
    embedding_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    max_retries: int = 3
) -> bool:
    """
    Store processed document in MongoDB with vector embeddings.
    
    Args:
        document: DoclingDocument instance
        original_filename: Original uploaded filename
        file_size: Size of original file in bytes
        mongodb_connection_string: MongoDB connection string
        database_name: Database name
        collection_name: Collection name
        embedding_config: Optional dict with embedding configuration
            - use_remote: bool (default: False for local)
            - model_name: str (default: "sentence-transformers/all-MiniLM-L6-v2")
            - api_key: str (required if use_remote=True)
        metrics: Optional metrics dictionary
        max_retries: Maximum number of retry attempts
    
    Returns:
        True if successful, False otherwise
    """
    if not PYMONGO_AVAILABLE:
        return False
    
    if embedding_config is None:
        embedding_config = {"use_remote": False, "model_name": DEFAULT_EMBEDDING_MODEL}
    
    try:
        app_config = get_settings()
        
        # Parse connection string to check format
        if not mongodb_connection_string.startswith('mongodb+srv://') and not mongodb_connection_string.startswith('mongodb://'):
            raise ValueError("Invalid MongoDB connection string format. Use mongodb+srv://...")

        # Mask connection string for logging
        masked_uri = app_config.mask_connection_string(mongodb_connection_string)
        logging.info(f"MongoDB: connecting with URI {masked_uri}")

        # Configure MongoClient with proper SSL/TLS settings
        client = MongoClient(
            mongodb_connection_string,
            tls=True,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=False,
            tlsAllowInvalidHostnames=False,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=20000,
            socketTimeoutMS=20000,
            retryWrites=True,
            retryReads=True,
            directConnection=False
        )

        # Test the connection before proceeding
        try:
            client.admin.command('ping')
            logging.info("MongoDB: connected successfully")
        except OperationFailure as auth_error:
            if auth_error.code == 8000:
                logging.error(
                    f"MongoDB auth failed (code 8000) with URI {masked_uri}. "
                    "Did the password change? If it contains special characters (@:/#?&% etc.), "
                    "URL-encode it using urllib.parse.quote_plus()."
                )
            raise

        db = client[database_name]
        collection = db[collection_name]
        
        # Chunk the document
        chunks_data = chunk_document(document)
        chunk_texts = [chunk["text"] for chunk in chunks_data]
        
        # Generate embeddings
        embeddings = generate_embeddings(chunk_texts, embedding_config)
        
        # Determine embedding dimensions and model name
        embedding_dim = len(embeddings[0]) if embeddings else 0
        model_name = embedding_config.get("model_name", DEFAULT_EMBEDDING_MODEL)
        
        # Combine chunks with embeddings
        chunks_with_embeddings = []
        for chunk_data, embedding in zip(chunks_data, embeddings):
            chunks_with_embeddings.append({
                "text": chunk_data["text"],
                "embedding": embedding,
                "chunk_index": chunk_data["chunk_index"],
                "metadata": chunk_data["metadata"]
            })
        
        # Prepare primary document for storage
        primary_doc = {
            "filename": original_filename,
            "original_filename": original_filename,
            "processed_at": datetime.utcnow().isoformat(),
            "file_size": file_size,
            "docling_json": document.export_to_dict(),
            "markdown": document.export_to_markdown(),
            "chunks": chunks_with_embeddings,
            "metadata": {
                "total_chunks": len(chunks_with_embeddings),
                "embedding_model": model_name,
                "embedding_dimensions": embedding_dim,
                "use_remote": embedding_config.get("use_remote", False)
            }
        }
        
        # Add quality metrics if provided
        if metrics:
            primary_doc.update({
                "input": metrics.get("input", {}),
                "metrics": metrics.get("metrics", {}),
                "warnings": metrics.get("warnings", {}),
                "ocr": metrics.get("ocr", {}),
                "status": metrics.get("status", {}),
                "text_layer_detected": metrics.get("text_layer_detected", False),
                "rasterized_graphics_skipped": metrics.get("rasterized_graphics_skipped", 0),
                "schema_version": QA_SCHEMA_VERSION,
            })
            
            # Add correlation IDs
            run_id = metrics.get("run_id")
            content_hash = metrics.get("content_hash")
            if run_id:
                primary_doc["run_id"] = run_id
            if content_hash:
                primary_doc["content_hash"] = content_hash

        # Sanitize document for MongoDB BSON compatibility
        primary_doc = sanitize_for_mongodb(primary_doc)
        
        # Helper function for exponential backoff retry
        def retry_with_backoff(operation, operation_name, max_retries=max_retries):
            """Retry operation with exponential backoff."""
            for attempt in range(max_retries):
                try:
                    return operation()
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    wait_time = 2 ** attempt
                    logging.warning(
                        f"MongoDB {operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
        
        # Check if we need to split due to size
        if bson_len(primary_doc) >= MAX_BSON_SAFE:
            # Extract pages from docling_json for separate storage
            docling_dict = primary_doc.get("docling_json", {})
            pages_raw = docling_dict.get("pages", [])
            
            # Convert pages to list of dicts for splitting
            page_docs_raw = []
            for i, page in enumerate(pages_raw):
                if isinstance(page, dict):
                    page_docs_raw.append(page)
                else:
                    page_docs_raw.append({"page_index": i, "content": str(page)})
            
            # Split primary doc from pages
            small_primary, page_docs = split_for_mongo(primary_doc, page_docs_raw)
            
            # Idempotent upsert by content_hash + filename
            content_hash = small_primary.get("content_hash")
            filename = small_primary.get("filename")
            if content_hash and filename:
                primary_result = retry_with_backoff(
                    lambda: collection.update_one(
                        {"content_hash": content_hash, "filename": filename},
                        {"$set": small_primary},
                        upsert=True
                    ),
                    "primary document upsert"
                )
                primary_id = primary_result.upserted_id if primary_result.upserted_id else collection.find_one(
                    {"content_hash": content_hash, "filename": filename}
                )["_id"]
            else:
                # Fallback to insert if content_hash missing
                primary_result = retry_with_backoff(
                    lambda: collection.insert_one(small_primary),
                    "primary document insert"
                )
                primary_id = primary_result.inserted_id
            
            # Insert pages into separate collection
            pages_collection = db[f"{collection_name}_pages"]
            for i, page_doc in enumerate(page_docs):
                page_doc["parent_id"] = primary_id
                page_doc["page_index"] = i
                retry_with_backoff(
                    lambda pd=page_doc: pages_collection.insert_one(pd),
                    f"page {i} insert"
                )
            
            logging.info(f"Document split: primary ({bson_len(small_primary)} bytes) + {len(page_docs)} pages")
        else:
            # Document fits in single BSON doc - idempotent upsert
            content_hash = primary_doc.get("content_hash")
            filename = primary_doc.get("filename")
            if content_hash and filename:
                retry_with_backoff(
                    lambda: collection.update_one(
                        {"content_hash": content_hash, "filename": filename},
                        {"$set": primary_doc},
                        upsert=True
                    ),
                    "document upsert"
                )
            else:
                # Fallback to insert if content_hash missing
                retry_with_backoff(
                    lambda: collection.insert_one(primary_doc),
                    "document insert"
                )
        
        # Create vector search index if it doesn't exist
        try:
            search_index_model = SearchIndexModel(
                definition={
                    "fields": [{
                        "type": "vector",
                        "path": "chunks.embedding",
                        "numDimensions": embedding_dim,
                        "similarity": "dotProduct"
                    }]
                },
                name="vector_index",
                type="vectorSearch"
            )
            collection.create_search_index(model=search_index_model)
        except Exception:
            # Index creation might fail if it already exists or lacks permissions
            pass
        
        return True
        
    except Exception as e:
        logging.error(f"MongoDB storage failed: {e}")
        return False

