#!/usr/bin/env python3
"""
Backfill script to add missing quality metrics fields to existing MongoDB documents.

This script adds minimal default values for missing keys:
- input, metrics, warnings, ocr, status
- text_layer_detected, rasterized_graphics_skipped
- schema_version=1 (for existing docs)

It does NOT overwrite existing fields.
"""

import os
import sys
from typing import Dict, Any

try:
    from pymongo import MongoClient
except ImportError:
    print("ERROR: pymongo not installed. Install with: pip install pymongo[srv]")
    sys.exit(1)


def get_mongo_client(connection_string: str):
    """Create MongoDB client with SSL/TLS configuration."""
    import certifi
    return MongoClient(
        connection_string,
        tls=True,
        tlsCAFile=certifi.where(),
        tlsAllowInvalidCertificates=False,
        tlsAllowInvalidHostnames=False,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000,
    )


def backfill_document(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add missing quality metrics fields to a document.
    
    Args:
        doc: MongoDB document dictionary
        
    Returns:
        Updated document dictionary (only if changes needed)
    """
    updated = False
    updates = {}
    
    # Add input if missing
    if "input" not in doc:
        updates["input"] = {"format": "unknown"}
        updated = True
    
    # Add metrics if missing
    if "metrics" not in doc:
        updates["metrics"] = {
            "page_count": None,
            "markdown_length": len(doc.get("markdown", "")),
            "process_seconds": None,
            "block_count": 0,
            "heading_count": 0,
            "table_count": 0,
            "figure_count": 0,
        }
        updated = True
    
    # Add warnings if missing
    if "warnings" not in doc:
        updates["warnings"] = {
            "wmf_missing_loader": False,
            "osd_fail_count": 0,
            "format_conflict": False,
        }
        updated = True
    
    # Add ocr if missing
    if "ocr" not in doc:
        updates["ocr"] = {"engine_used": "unknown"}
        updated = True
    
    # Add status if missing
    if "status" not in doc:
        updates["status"] = {"quality_bucket": "ok"}
        updated = True
    
    # Add text_layer_detected if missing
    if "text_layer_detected" not in doc:
        updates["text_layer_detected"] = False
        updated = True
    
    # Add rasterized_graphics_skipped if missing
    if "rasterized_graphics_skipped" not in doc:
        updates["rasterized_graphics_skipped"] = 0
        updated = True
    
    # Add schema_version if missing (set to 1 for existing docs)
    if "schema_version" not in doc:
        updates["schema_version"] = 1
        updated = True
    
    if updated:
        doc.update(updates)
        return updates
    return None


def main():
    """Main backfill function."""
    # Get MongoDB connection from environment
    connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    database_name = os.getenv("MONGODB_DATABASE", "docling_db")
    collection_name = os.getenv("MONGODB_COLLECTION", "documents")
    
    if not connection_string:
        print("ERROR: MONGODB_CONNECTION_STRING environment variable not set")
        print("Usage: MONGODB_CONNECTION_STRING='mongodb+srv://...' python scripts/backfill_min_metrics.py")
        sys.exit(1)
    
    print(f"Connecting to MongoDB: {database_name}/{collection_name}")
    
    try:
        client = get_mongo_client(connection_string)
        db = client[database_name]
        collection = db[collection_name]
        
        # Test connection
        client.admin.command('ping')
        print("✓ Connected to MongoDB")
        
        # Find all documents
        total_docs = collection.count_documents({})
        print(f"Found {total_docs} documents to check")
        
        updated_count = 0
        skipped_count = 0
        
        # Process documents in batches
        batch_size = 100
        for i in range(0, total_docs, batch_size):
            docs = list(collection.find({}).skip(i).limit(batch_size))
            
            for doc in docs:
                doc_id = doc.get("_id")
                updates = backfill_document(doc)
                
                if updates:
                    # Update document in MongoDB
                    collection.update_one(
                        {"_id": doc_id},
                        {"$set": updates}
                    )
                    updated_count += 1
                    if updated_count % 10 == 0:
                        print(f"  Updated {updated_count} documents...")
                else:
                    skipped_count += 1
        
        print(f"\n✓ Backfill complete:")
        print(f"  - Updated: {updated_count} documents")
        print(f"  - Skipped (already complete): {skipped_count} documents")
        print(f"  - Total: {total_docs} documents")
        
        client.close()
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

