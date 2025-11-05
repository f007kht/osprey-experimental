"""
MongoDB helper functions for handling large documents and BSON size limits.

MongoDB has a 16MB BSON document size limit. This module provides utilities
to split large documents into smaller pieces and manage document size.
"""

import json
from typing import Dict, List, Any, Tuple, Mapping

try:
    import bson
    BSON_AVAILABLE = True
except ImportError:
    BSON_AVAILABLE = False

# 16 MB hard limit; keep ~200 KB headroom for envelope/metadata to avoid edge cases
MAX_BSON_BYTES = 16 * 1024 * 1024
BSON_HEADROOM_BYTES = 200 * 1024
MAX_BSON_SAFE = MAX_BSON_BYTES - BSON_HEADROOM_BYTES

# Optional: allow tuning via env without risking accidental overshoot
try:
    import os
    _env_headroom = int(os.getenv("BSON_HEADROOM_BYTES", str(BSON_HEADROOM_BYTES)))
    if 0 < _env_headroom < MAX_BSON_BYTES:
        BSON_HEADROOM_BYTES = _env_headroom
        MAX_BSON_SAFE = MAX_BSON_BYTES - BSON_HEADROOM_BYTES
except Exception:
    pass

# Backward compatibility alias
MAX_BSON = MAX_BSON_SAFE


def bson_len(doc: dict) -> int:
    """
    Calculate the approximate BSON size of a document.
    
    Args:
        doc: Dictionary to measure
        
    Returns:
        Approximate size in bytes
    """
    if not BSON_AVAILABLE:
        # Rough proxy if bson not available - use JSON encoding
        try:
            return len(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
        except Exception:
            # Fallback: estimate based on string representation
            return len(str(doc).encode("utf-8"))
    
    try:
        encoded = bson.BSON.encode(doc)
        # Size guard to fail-fast before insert
        if len(encoded) > MAX_BSON_SAFE:
            raise ValueError(
                f"BSON document exceeds safe size ({len(encoded)} bytes > {MAX_BSON_SAFE} bytes). "
                f"Consider chunking/downsizing payload."
            )
        return len(encoded)
    except ValueError:
        # Re-raise size guard errors
        raise
    except Exception:
        # If encoding fails, use JSON as fallback
        try:
            return len(json.dumps(doc, ensure_ascii=False).encode("utf-8"))
        except Exception:
            # Last resort: string length
            return len(str(doc).encode("utf-8"))


def split_for_mongo(primary: dict, pages: List[dict]) -> Tuple[dict, List[dict]]:
    """
    Ensure primary doc < 16MB; drop heavy fields to page docs; tag truncation.
    
    Moves heavy fields (images, tokens, full_export, chunks with embeddings) out
    of the primary document if it exceeds the BSON limit. Pages are also
    checked and truncated if needed.
    
    Args:
        primary: Primary document dictionary
        pages: List of page document dictionaries
        
    Returns:
        Tuple of (small_primary, page_docs)
    """
    small = dict(primary)
    
    # Move heavy fields out of primary if needed
    # These fields are typically large and can cause BSON limit issues
    heavy_fields = ["images", "tokens", "full_export", "chunks", "embeddings"]
    
    # Check primary doc size and remove heavy fields if needed
    if bson_len(small) >= MAX_BSON_SAFE:
        # Remove heavy fields one by one until under limit
        for field in heavy_fields:
            if field in small:
                small.pop(field, None)
                if bson_len(small) < MAX_BSON_SAFE:
                    break
                # Mark that heavy data was removed
                small["_heavy_fields_removed"] = heavy_fields[:heavy_fields.index(field) + 1]
    
    # Process pages
    page_docs = []
    for i, page in enumerate(pages):
        p = dict(page)
        
        # Check if page exceeds limit
        if bson_len(p) >= MAX_BSON_SAFE:
            # Last resort: drop images and mark as truncated
            if "images" in p:
                p.pop("images", None)
            p["_truncated"] = True
            p["_truncation_reason"] = "exceeded_bson_limit"
        
        page_docs.append(p)
    
    return small, page_docs


def should_split_document(doc: dict, threshold: int = None) -> bool:
    """
    Check if a document should be split due to size.
    
    Args:
        doc: Document to check
        threshold: Size threshold in bytes (default: MAX_BSON_SAFE)
        
    Returns:
        True if document should be split
    """
    if threshold is None:
        threshold = MAX_BSON_SAFE
    
    return bson_len(doc) >= threshold

