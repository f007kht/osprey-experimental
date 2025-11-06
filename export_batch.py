#!/usr/bin/env python3
"""
Batch export MongoDB documents by ObjectId list.

Usage:
    python export_batch.py 690925b1306fdd023888fae4 690846385135ea54282d86b0 ...
    
Or with connection string:
    MONGODB_CONNECTION_STRING='...' python export_batch.py <id1> <id2> ...
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional

try:
    from pymongo import MongoClient
    from bson import ObjectId, json_util
    import certifi
    PYMONGO_AVAILABLE = True
except ImportError:
    print("ERROR: pymongo not available. Install with: pip install pymongo[srv]")
    sys.exit(1)

# Try to load app_settings for connection string
try:
    from app_settings import get_settings
    app_config = get_settings()
    DEFAULT_CONNECTION_STRING = app_config.mongodb_connection_string
    # Check both docling_db and docling_documents (documents may be in either)
    DEFAULT_DB = os.getenv("MONGODB_DATABASE", "docling_db")  # Default to docling_db
    DEFAULT_COLL = app_config.mongodb_collection
except ImportError:
    DEFAULT_CONNECTION_STRING = None
    DEFAULT_DB = os.getenv("MONGODB_DATABASE", "docling_db")  # Default to docling_db
    DEFAULT_COLL = os.getenv("MONGODB_COLLECTION", "documents")

DEFAULT_EXPORT_DIR = os.getenv("MONGODB_EXPORT_DIR", r"D:\.osprey\mongodb_exports")


def get_mongo_client(connection_string: Optional[str] = None) -> MongoClient:
    """Get MongoDB client from connection string."""
    if connection_string is None:
        connection_string = os.environ.get("MONGODB_CONNECTION_STRING") or DEFAULT_CONNECTION_STRING
        if not connection_string:
            raise ValueError(
                "MongoDB connection string required "
                "(set MONGODB_CONNECTION_STRING env var or configure in app_settings)"
            )
    
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


def export_documents(ids: List[str], output_dir: Optional[str] = None, pretty: bool = True, database: Optional[str] = None) -> None:
    """Export multiple documents by ObjectId."""
    if not ids:
        print("ERROR: No document IDs provided", file=sys.stderr)
        sys.exit(1)
    
    # Get MongoDB client
    try:
        client = get_mongo_client()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output directory
    if output_dir is None:
        output_dir = DEFAULT_EXPORT_DIR
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use specified database or default
    db_name = database or DEFAULT_DB
    db = client[db_name]
    collection = db[DEFAULT_COLL]
    
    print(f"Using database: {db_name}, collection: {DEFAULT_COLL}")
    
    exported = 0
    not_found = []
    
    print(f"Exporting {len(ids)} document(s) to {output_dir}...")
    print("=" * 60)
    
    for doc_id in ids:
        try:
            # Try to convert to ObjectId
            obj_id = ObjectId(doc_id) if ObjectId.is_valid(doc_id) else doc_id
            
            # Find document
            doc = collection.find_one({"_id": obj_id})
            
            if not doc:
                print(f"[WARN] Not found: {doc_id}")
                not_found.append(doc_id)
                continue
            
            # Generate output filename
            filename = f"doc_{doc_id}.json"
            output_path = os.path.join(output_dir, filename)
            
            # Export to JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json_str = json_util.dumps(doc, indent=2 if pretty else None)
                f.write(json_str)
            
            # Get document info
            filename_doc = doc.get("filename") or doc.get("original_filename", "unknown")
            format_doc = doc.get("input", {}).get("format", "unknown")
            processed = doc.get("processed_at", "unknown")
            
            print(f"[OK] Exported: {doc_id}")
            print(f"  File: {filename_doc}")
            print(f"  Format: {format_doc}")
            print(f"  Processed: {processed}")
            print(f"  Output: {output_path}")
            print()
            
            exported += 1
            
        except Exception as e:
            print(f"[ERROR] Error exporting {doc_id}: {e}", file=sys.stderr)
            not_found.append(doc_id)
    
    client.close()
    
    print("=" * 60)
    print(f"Summary: {exported} exported, {len(not_found)} not found")
    
    if not_found:
        print(f"\nNot found IDs: {', '.join(not_found)}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch export MongoDB documents by ObjectId")
    parser.add_argument("ids", nargs="+", help="Document ObjectIds to export")
    parser.add_argument("--db", help="Database name (default: docling_db)", default=None)
    parser.add_argument("--output-dir", help="Output directory", default=None)
    parser.add_argument("--no-pretty", action="store_true", help="Disable pretty printing")
    
    args = parser.parse_args()
    
    export_documents(
        args.ids,
        output_dir=args.output_dir,
        pretty=not args.no_pretty,
        database=args.db
    )


if __name__ == "__main__":
    main()

