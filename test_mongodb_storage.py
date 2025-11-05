"""
Test script to verify MongoDB storage of processed documents.

This script helps verify that documents were stored correctly in MongoDB,
including checking for splits, formulas, pictures, and embeddings.
"""

import os
import sys
import collections
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from pymongo import MongoClient
    PYMONGO_AVAILABLE = True
except ImportError:
    print("ERROR: pymongo not available. Install with: pip install pymongo[srv]")
    sys.exit(1)


def get_mongo_client(connection_string: Optional[str] = None) -> MongoClient:
    """Get MongoDB client from connection string."""
    if connection_string is None:
        connection_string = os.environ.get("MONGODB_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("MongoDB connection string required (set MONGODB_CONNECTION_STRING env var)")
    
    import certifi
    return MongoClient(
        connection_string,
        tls=True,
        tlsCAFile=certifi.where(),
        tlsAllowInvalidCertificates=False,
        tlsAllowInvalidHostnames=False,
    )


def walk_counts(node: Any, ctr: collections.Counter) -> None:
    """
    Recursively walk a document structure and count node types/labels.
    
    Args:
        node: Document node (dict, list, or primitive)
        ctr: Counter to accumulate counts
    """
    if isinstance(node, dict):
        label = (node.get("label") or node.get("type") or "").lower()
        if label:
            ctr[label] += 1
        for v in node.values():
            walk_counts(v, ctr)
    elif isinstance(node, list):
        for x in node:
            walk_counts(x, ctr)


def analyze_document(
    db,
    collection_name: str,
    filename: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze a stored document in MongoDB.
    
    Args:
        db: MongoDB database instance
        collection_name: Name of the primary collection
        filename: Original filename to search for
        verbose: Print detailed information
        
    Returns:
        Dictionary with analysis results
    """
    collection = db[collection_name]
    pages_collection = db[f"{collection_name}_pages"]
    
    # Find primary document
    doc = collection.find_one({"original_filename": filename})
    if not doc:
        if verbose:
            print(f"ERROR: Document not found: {filename}")
        return {"error": "document_not_found"}
    
    results = {
        "filename": doc.get("original_filename"),
        "file_size": doc.get("file_size"),
        "processed_at": doc.get("processed_at"),
        "total_chunks": doc.get("metadata", {}).get("total_chunks", 0),
        "embedding_model": doc.get("metadata", {}).get("embedding_model"),
        "embedding_dimensions": doc.get("metadata", {}).get("embedding_dimensions"),
        "primary_id": str(doc["_id"]),
    }
    
    # Check for page splits
    page_count = pages_collection.count_documents({"parent_id": doc["_id"]})
    results["pages_count"] = page_count
    
    if page_count > 0:
        results["was_split"] = True
        truncated_pages = pages_collection.count_documents({
            "parent_id": doc["_id"],
            "_truncated": True
        })
        results["truncated_pages"] = truncated_pages
    else:
        results["was_split"] = False
        results["truncated_pages"] = 0
    
    # Analyze docling_json structure
    docling_json = doc.get("docling_json", {})
    if docling_json:
        counts = collections.Counter()
        walk_counts(docling_json, counts)
        results["node_counts"] = dict(counts.most_common(20))
        
        # Check for key elements
        results["has_formulas"] = any(
            "formula" in k.lower() for k in counts.keys()
        )
        results["has_pictures"] = any(
            "picture" in k.lower() for k in counts.keys()
        )
        results["has_tables"] = any(
            "table" in k.lower() for k in counts.keys()
        )
    
    # Print results if verbose
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analysis for: {filename}")
        print(f"{'='*60}")
        print(f"File size: {results['file_size']:,} bytes")
        print(f"Processed at: {results['processed_at']}")
        print(f"Total chunks: {results['total_chunks']}")
        print(f"Embedding model: {results['embedding_model']}")
        print(f"Embedding dimensions: {results['embedding_dimensions']}")
        print(f"\nStorage:")
        print(f"  Was split: {results['was_split']}")
        if results['was_split']:
            print(f"  Pages in separate collection: {results['pages_count']}")
            if results['truncated_pages'] > 0:
                print(f"  ⚠️  Truncated pages: {results['truncated_pages']}")
        
        print(f"\nDocument structure (top nodes):")
        for node_type, count in list(results.get("node_counts", {}).items())[:12]:
            print(f"  {node_type}: {count}")
        
        print(f"\nContent detection:")
        print(f"  Formulas: {'✓' if results.get('has_formulas') else '✗'}")
        print(f"  Pictures: {'✓' if results.get('has_pictures') else '✗'}")
        print(f"  Tables: {'✓' if results.get('has_tables') else '✗'}")
        print(f"{'='*60}\n")
    
    return results


def create_indexes(db, collection_name: str, verbose: bool = True) -> None:
    """
    Create recommended indexes for MongoDB collections.
    
    Args:
        db: MongoDB database instance
        collection_name: Name of the primary collection
        verbose: Print index creation status
    """
    collection = db[collection_name]
    pages_collection = db[f"{collection_name}_pages"]
    
    indexes_created = []
    
    try:
        collection.create_index("processed_at", background=True)
        indexes_created.append(f"{collection_name}.processed_at")
    except Exception as e:
        if verbose:
            print(f"Index creation note: {collection_name}.processed_at - {e}")
    
    try:
        pages_collection.create_index("parent_id", background=True)
        indexes_created.append(f"{collection_name}_pages.parent_id")
    except Exception as e:
        if verbose:
            print(f"Index creation note: {collection_name}_pages.parent_id - {e}")
    
    try:
        pages_collection.create_index(
            [("parent_id", 1), ("page_index", 1)],
            unique=True,
            background=True
        )
        indexes_created.append(f"{collection_name}_pages.parent_id+page_index (unique)")
    except Exception as e:
        if verbose:
            print(f"Index creation note: {collection_name}_pages.parent_id+page_index - {e}")
    
    if verbose:
        print(f"Created indexes: {', '.join(indexes_created)}")


def list_recent_documents(
    db,
    collection_name: str,
    limit: int = 10,
    verbose: bool = True
) -> list:
    """
    List recent documents in MongoDB.
    
    Args:
        db: MongoDB database instance
        collection_name: Name of the collection
        limit: Maximum number of documents to return
        verbose: Print document list
        
    Returns:
        List of document dictionaries
    """
    collection = db[collection_name]
    docs = list(collection.find(
        {},
        {
            "original_filename": 1,
            "file_size": 1,
            "processed_at": 1,
            "metadata.total_chunks": 1,
        }
    ).sort("processed_at", -1).limit(limit))
    
    if verbose:
        print(f"\nRecent documents (last {limit}):")
        print(f"{'='*60}")
        for doc in docs:
            print(f"  {doc.get('original_filename', 'unknown')}")
            print(f"    Size: {doc.get('file_size', 0):,} bytes")
            print(f"    Processed: {doc.get('processed_at', 'unknown')}")
            print(f"    Chunks: {doc.get('metadata', {}).get('total_chunks', 0)}")
            print()
    
    return docs


def main():
    """Main entry point for the test script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test and analyze MongoDB storage of processed documents"
    )
    parser.add_argument(
        "--connection-string",
        help="MongoDB connection string (or set MONGODB_CONNECTION_STRING env var)"
    )
    parser.add_argument(
        "--database",
        default="docling_documents",
        help="Database name (default: docling_documents)"
    )
    parser.add_argument(
        "--collection",
        default="documents",
        help="Collection name (default: documents)"
    )
    parser.add_argument(
        "--create-indexes",
        action="store_true",
        help="Create recommended indexes"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent documents"
    )
    parser.add_argument(
        "--analyze",
        help="Analyze a specific document by filename"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit for list operation (default: 10)"
    )
    
    args = parser.parse_args()
    
    try:
        client = get_mongo_client(args.connection_string)
        db = client[args.database]
        
        if args.create_indexes:
            print("Creating indexes...")
            create_indexes(db, args.collection)
        
        if args.list:
            list_recent_documents(db, args.collection, limit=args.limit)
        
        if args.analyze:
            analyze_document(db, args.collection, args.analyze)
        
        if not (args.create_indexes or args.list or args.analyze):
            parser.print_help()
            print("\nExample usage:")
            print("  python test_mongodb_storage.py --create-indexes")
            print("  python test_mongodb_storage.py --list")
            print("  python test_mongodb_storage.py --analyze 'document.pdf'")
        
        client.close()
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

