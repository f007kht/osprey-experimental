#!/usr/bin/env python3
"""
Script to check if images were processed and stored correctly.

This script queries MongoDB to inspect the image data from processed documents.
It shows:
- Whether page images were generated
- Whether picture images were extracted
- Image metadata (dimensions, format, etc.)
- Sample image data (base64 preview)

Usage:
    # Check the most recent document
    python scripts/check_images.py

    # Check a specific document by run_id
    python scripts/check_images.py --run-id 8af580e5-825a-4281-b72d-599e3603aeb2

    # Check by content hash (first 8 chars)
    python scripts/check_images.py --hash b8a3ded9
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_settings import get_settings


def format_bytes(size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def check_document_images(doc: Dict[str, Any]) -> None:
    """
    Check and display image information from a document.

    Args:
        doc: Document dictionary from MongoDB
    """
    print("\n" + "="*80)
    print("DOCUMENT IMAGE ANALYSIS")
    print("="*80)

    # Document metadata
    print(f"\n📄 Document: {doc.get('input', {}).get('filename', 'unknown')}")
    print(f"   Run ID: {doc.get('run_id', 'N/A')}")
    print(f"   Hash: {doc.get('content_hash', 'N/A')[:8]}")
    print(f"   Format: {doc.get('input', {}).get('format', 'N/A')}")
    print(f"   Pages: {doc.get('metrics', {}).get('page_count', 'N/A')}")
    print(f"   Processed: {doc.get('input', {}).get('upload_timestamp', 'N/A')}")

    # Check if docling_json exists
    docling_json = doc.get('docling_json')
    if not docling_json:
        print("\n❌ No docling_json data found in document")
        print("   This document may not have image data stored.")
        return

    # Check for pictures in the document
    pictures = docling_json.get('pictures', [])
    print(f"\n🖼️  Pictures Found: {len(pictures)}")

    if pictures:
        print("\nPicture Details:")
        for idx, picture in enumerate(pictures, 1):
            print(f"\n  Picture #{idx}:")

            # Picture metadata
            prov = picture.get('prov', [{}])[0] if picture.get('prov') else {}
            print(f"    Page: {prov.get('page_no', 'N/A')}")
            print(f"    Bounding Box: {prov.get('bbox', 'N/A')}")

            # Check for image data
            data = picture.get('data', {})
            if data:
                # Check for base64 encoded image
                image_str = data.get('image', '')
                if image_str:
                    # Image is stored as base64 string
                    print(f"    ✅ Image Data: Yes (base64 encoded)")
                    print(f"    Size: ~{format_bytes(len(image_str))}")

                    # Try to detect image format from base64 prefix
                    if image_str.startswith('/9j/'):
                        print(f"    Format: JPEG")
                    elif image_str.startswith('iVBOR'):
                        print(f"    Format: PNG")
                    else:
                        print(f"    Format: Unknown")

                    # Show first 50 chars of base64 as preview
                    print(f"    Preview: {image_str[:50]}...")
                else:
                    print(f"    ❌ Image Data: No base64 data found")

                # Check for image dimensions
                if 'width' in data or 'height' in data:
                    print(f"    Dimensions: {data.get('width', '?')} × {data.get('height', '?')}")
            else:
                print(f"    ❌ No image data field found")

            # Check caption
            caption = picture.get('text', '')
            if caption:
                print(f"    Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
    else:
        print("   No pictures found in document.")
        print("   This could mean:")
        print("   - The PDF had no embedded images")
        print("   - Image extraction was disabled")
        print("   - Images failed to extract")

    # Check for page images in main_text
    main_text = docling_json.get('main_text', [])
    page_images_found = 0

    for item in main_text:
        if item.get('type') == 'page':
            # Check if page has image data
            data = item.get('data', {})
            if 'image' in data and data['image']:
                page_images_found += 1

    print(f"\n📄 Page Images: {page_images_found}")
    if page_images_found > 0:
        print(f"   ✅ Full-page images were generated")
        print(f"   These are complete page renders at configured scale")
    else:
        print(f"   ℹ️  No full-page images found")
        print(f"   This is expected if IMAGES_ENABLE=false or generate_page_images=false")

    # Image generation settings (if available in metadata)
    settings_summary = doc.get('settings_summary', {})
    if settings_summary:
        print("\n⚙️  Image Settings Used:")
        print(f"   Images Enabled: {settings_summary.get('images_enable', 'N/A')}")
        print(f"   Image Scale: {settings_summary.get('images_scale', 'N/A')}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    total_images = len(pictures) + page_images_found
    if total_images > 0:
        print(f"✅ Image processing appears to be working!")
        print(f"   Total images found: {total_images}")
        print(f"   - Picture images: {len(pictures)}")
        print(f"   - Page images: {page_images_found}")
    else:
        print(f"⚠️  No images found in this document")
        print(f"   Possible reasons:")
        print(f"   1. IMAGES_ENABLE environment variable is set to false")
        print(f"   2. The document had no embedded images")
        print(f"   3. Image extraction failed (check logs for errors)")

    print("="*80 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check if images were processed correctly in MongoDB documents"
    )
    parser.add_argument(
        "--run-id",
        help="Filter by run_id (UUID)"
    )
    parser.add_argument(
        "--hash",
        help="Filter by content_hash prefix (first 8 chars)"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        default=True,
        help="Check the most recent document (default)"
    )

    args = parser.parse_args()

    # Get settings
    settings = get_settings()

    if not settings.enable_mongodb:
        print("❌ MongoDB is not enabled")
        print("   Set ENABLE_MONGODB=true in environment variables")
        sys.exit(1)

    if not settings.mongodb_connection_string:
        print("❌ MongoDB connection string not configured")
        print("   Set MONGODB_CONNECTION_STRING in environment variables")
        sys.exit(1)

    # Connect to MongoDB
    try:
        from pymongo import MongoClient
        import certifi

        print(f"Connecting to MongoDB...")
        client = MongoClient(
            settings.mongodb_connection_string,
            serverSelectionTimeoutMS=5000,
            tls=True,
            tlsCAFile=certifi.where(),
        )

        db = client[settings.mongodb_database]
        collection = db[settings.mongodb_collection]

        # Query for document
        query = {}
        if args.run_id:
            query['run_id'] = args.run_id
            print(f"Searching for run_id: {args.run_id}")
        elif args.hash:
            # Match content_hash prefix
            import re
            query['content_hash'] = re.compile(f"^{args.hash}")
            print(f"Searching for content_hash starting with: {args.hash}")
        else:
            print("Fetching most recent document...")

        # Find document
        if query:
            doc = collection.find_one(query)
        else:
            # Get most recent document
            doc = collection.find_one(sort=[('input.upload_timestamp', -1)])

        if not doc:
            print("❌ No document found matching criteria")
            sys.exit(1)

        # Check images in document
        check_document_images(doc)

        client.close()

    except ImportError:
        print("❌ pymongo not installed")
        print("   Install with: pip install pymongo certifi")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
