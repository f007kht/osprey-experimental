"""Quality metrics extraction and assignment.

Module 1 - Osprey Backend: Document processing application.
"""

from typing import Dict, Any, Optional


def extract_document_metrics(
    result,
    markdown_text: str,
    fmt: str,
    process_seconds: float,
    extras: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Extract comprehensive metrics from conversion result.
    
    Args:
        result: ConversionResult from Docling (may be None if aborted)
        markdown_text: Exported markdown text
        fmt: Detected input format
        process_seconds: Processing time in seconds
        extras: Dictionary with additional metrics (warnings, ocr_engine, etc.)
        
    Returns:
        Dictionary with all metrics organized by category
    """
    # Extract page count (null-safe access for schema drift safety)
    page_count = None
    if result is not None and hasattr(result, 'document') and result.document:
        if hasattr(result.document, 'pages') and result.document.pages:
            page_count = len(result.document.pages)
        elif hasattr(result.document, 'page_count'):
            page_count = result.document.page_count
    
    d = {
        "input": {"format": fmt},
        "metrics": {
            "page_count": page_count,
            "markdown_length": len(markdown_text or ""),
            "process_seconds": round(process_seconds, 3),
            "block_count": 0,
            "heading_count": 0,
            "table_count": 0,
            "figure_count": 0,
        },
        "warnings": {
            "wmf_missing_loader": extras.get("warnings", {}).get("wmf_missing_loader", False),
            "osd_fail_count": extras.get("warnings", {}).get("osd_fail_count", 0),
            "format_conflict": extras.get("warnings", {}).get("format_conflict", False),
        },
        "ocr": {
            "engine_used": extras.get("ocr_engine", "unknown")
        },
        "text_layer_detected": extras.get("text_layer_detected", False),
        "rasterized_graphics_skipped": extras.get("rasterized_graphics_skipped", 0),
        "status": {"quality_bucket": "ok"},
    }
    
    # Handle abort status (null-safe access)
    if extras.get("abort"):
        d["status"]["abort"] = extras["abort"]
        d["status"]["quality_bucket"] = "suspect"
    
    # Try Docling dict export to count structures (null-safe)
    try:
        if result is not None and hasattr(result, 'document') and result.document:
            doc_dict = result.document.export_to_dict()
            blocks = doc_dict.get('blocks') or []
            d["metrics"]["block_count"] = len(blocks)
            d["metrics"]["heading_count"] = sum(1 for b in blocks if b.get('type') in ('heading', 'title', 'h1', 'h2', 'h3'))
            d["metrics"]["table_count"] = sum(1 for b in blocks if b.get('type') == 'table')
            d["metrics"]["figure_count"] = sum(1 for b in blocks if b.get('type') in ('figure', 'image', 'picture'))
    except Exception:
        # Fallback: attempt best-effort extraction from result.document.blocks
        try:
            if result is not None:
                doc = getattr(result, "document", None)
                blocks = getattr(doc, "blocks", []) or []
                if blocks:
                    d["metrics"]["block_count"] = len(blocks)
                    # Best-effort type detection if blocks have type attribute
                    d["metrics"]["heading_count"] = sum(1 for b in blocks if getattr(b, 'type', None) in ('heading', 'title', 'h1', 'h2', 'h3'))
                    d["metrics"]["table_count"] = sum(1 for b in blocks if getattr(b, 'type', None) == 'table')
                    d["metrics"]["figure_count"] = sum(1 for b in blocks if getattr(b, 'type', None) in ('figure', 'image', 'picture'))
        except Exception:
            pass
    
    # Only assign quality bucket if not already set by abort
    if not d["status"].get("abort"):
        d["status"]["quality_bucket"] = assign_quality_bucket(d)
    return d


def assign_quality_bucket(m: Dict[str, Any]) -> str:
    """
    Assign quality bucket (ok/suspect/fail) based on metrics.
    
    Args:
        m: Metrics dictionary from extract_document_metrics
        
    Returns:
        Quality bucket string: "ok", "suspect", or "fail"
    """
    md_len = m["metrics"]["markdown_length"]
    osd_fail = m["warnings"]["osd_fail_count"]
    tlayer = bool(m.get("text_layer_detected"))
    wmf = bool(m["warnings"]["wmf_missing_loader"])
    blocks = m["metrics"]["block_count"]
    tables = m["metrics"]["table_count"]
    figures = m["metrics"]["figure_count"]
    fmt = m["input"]["format"]
    
    # Fail: truly empty extract
    if blocks == 0 and tables == 0 and figures == 0 and md_len == 0:
        return "fail"
    
    # Suspect rules
    if md_len < 500:
        return "suspect"
    if fmt == "pdf" and osd_fail > 10 and not tlayer:
        return "suspect"
    if wmf:
        return "suspect"
    if md_len > 2_000_000:  # runaway duplication
        return "suspect"
    
    # Add note for PDFs with text layer but OSD failures (keep ok bucket but flag for dashboard)
    if fmt == "pdf" and tlayer and osd_fail > 0:
        m["status"]["notes"] = m["status"].get("notes", [])
        if "OSD_FAILS_ON_TEXTLAYER" not in m["status"]["notes"]:
            m["status"]["notes"].append("OSD_FAILS_ON_TEXTLAYER")
    
    return "ok"


def add_osd_collapsed_note(metrics: Dict[str, Any], extras: Dict[str, Any]) -> None:
    """Add OSD_COLLAPSED note to metrics if OSD was suppressed.
    
    Args:
        metrics: Metrics dictionary to update
        extras: Extras dictionary with suppression flags
    """
    if extras.get("osd_suppressed_after_first"):
        metrics["status"].setdefault("notes", []).append("OSD_COLLAPSED")

