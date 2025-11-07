"""Logging utilities for secret scrubbing and noise filtering.

Module 1 - Osprey Backend: Document processing application.
"""

import logging
import re
import warnings
from contextlib import contextmanager
from typing import Dict, Any


# Secret scrubbing regex
SECRET_URI_RE = re.compile(r"(mongodb\+srv://|mongodb://)([^:@/]+):([^@/]+)@")


def scrub_secrets(msg: str) -> str:
    """Scrub secrets (MongoDB connection strings, etc.) from log messages.
    
    Args:
        msg: Log message that may contain secrets
        
    Returns:
        Message with secrets redacted
    """
    if not msg:
        return msg
    return SECRET_URI_RE.sub(r"\1***:***@", msg)


class PdfNoiseFilter(logging.Filter):
    """Filter to block PDF noise messages (PK/EOF warnings) from non-PDF flows."""
    
    BLOCK_PATTERNS = (
        "invalid pdf header: b'PK\\x03\\x04\\x14'",
        "EOF marker not found",
    )
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out PDF noise messages."""
        msg = str(record.getMessage() or "")
        return not any(p in msg for p in self.BLOCK_PATTERNS)


# Singleton instance for reuse
_pdf_noise_filter = PdfNoiseFilter()


@contextmanager
def suppress_pdf_noise_for_non_pdf(active: bool):
    """
    Context manager to suppress PDF header/EOF warnings at logger level for non-PDF files.
    
    This ensures magic-byte sniffing happens first, and PDF library probes don't emit
    noise during non-PDF processing.
    
    Args:
        active: If True, suppress warnings; if False, pass through unchanged
    """
    if not active:
        yield
        return
    
    loggers = [
        logging.getLogger("pdfminer"),
        logging.getLogger("pypdf"),
        logging.getLogger("docling"),
        logging.getLogger("docling.pdf"),
        logging.getLogger("docling_core"),
    ]
    
    try:
        # Add filter to all PDF-related loggers
        for lg in loggers:
            lg.addFilter(_pdf_noise_filter)
        # Also catch warnings module
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r"invalid pdf header: b'PK\\x03\\x04\\x14'")
            warnings.filterwarnings("ignore", message=r"EOF marker not found")
            yield
    finally:
        # Remove filter from all loggers
        for lg in loggers:
            try:
                lg.removeFilter(_pdf_noise_filter)
            except Exception:
                pass


class PerDocOsdFilter(logging.Filter):
    """Per-document OSD filter for interleaving safety (matches by temp filename)."""
    
    def __init__(self, doc_token: str, extras: Dict[str, Any]):
        """Initialize filter with document token and extras dict.
        
        Args:
            doc_token: Temporary filename string to match in log messages
            extras: Dictionary to store suppression flags
        """
        super().__init__()
        self.doc_token = doc_token
        self.extras = extras
        self.hit = False
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter OSD messages: allow first, suppress subsequent for this document.
        
        Args:
            record: Log record to filter
            
        Returns:
            True to allow message, False to suppress
        """
        msg = str(record.getMessage() or "")
        if "OSD failed for doc" in msg and self.doc_token in msg:
            if not self.hit:
                self.hit = True
                self.extras["osd_suppressed_after_first"] = True
                # Allow the first one through
                return True
            # Swallow subsequent messages for THIS doc only
            return False
        return True


class MetricsLogHandler(logging.Handler):
    """Log handler for tracking metrics (OSD failures, WMF warnings, etc.)."""
    
    def __init__(self, extras_dict: Dict[str, Any]):
        """Initialize handler with extras dictionary for metrics storage.
        
        Args:
            extras_dict: Dictionary to store tracked metrics
        """
        super().__init__()
        self.extras = extras_dict
        # Reset counters per run to avoid cross-doc leakage
        self.extras.setdefault("warnings", {})["osd_fail_count"] = 0
    
    def emit(self, record: logging.LogRecord):
        """Process log record and update metrics.
        
        Args:
            record: Log record to process
        """
        msg = self.format(record)
        # Scrub secrets from log messages
        msg_scrubbed = scrub_secrets(msg)
        
        # Track OSD failures (count only - suppression handled by PerDocOsdFilter)
        if 'OSD failed' in msg_scrubbed:
            warnings_dict = self.extras.setdefault("warnings", {})
            # Increment count (always count, even if suppressed by filter)
            warnings_dict["osd_fail_count"] = warnings_dict.get("osd_fail_count", 0) + 1
        
        # Track WMF/EMF warnings (will be moved to text_processing module)
        if 'WMF' in msg_scrubbed or 'EMF' in msg_scrubbed or 'cannot be loaded by Pillow' in msg_scrubbed:
            warnings_dict = self.extras.setdefault("warnings", {})
            if "wmf_missing_loader" not in warnings_dict:
                warnings_dict["wmf_missing_loader"] = True
        
        # Note: This handler is for tracking only; actual log emission happens via root logger

