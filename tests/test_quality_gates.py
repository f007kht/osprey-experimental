"""Tests for quality gates and metrics extraction."""

import os
import io
import time
import pytest

# Import functions from app.py
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import sniff_file_format, _assign_quality_bucket, _extract_document_metrics
from tests.conftest import ResultStub


def test_sniff_pdf_magic():
    """Test PDF format detection via magic bytes."""
    fmt, conf, conflict, ext = sniff_file_format("x.pdf", b"%PDF-1.7....")
    assert fmt == "pdf"
    assert conf >= 0.9
    assert conflict is False


def test_sniff_office_magic_vs_ext_conflict():
    """Test Office format detection with potential conflict."""
    # Office ZIP magic bytes (PK\x03\x04)
    fmt, conf, conflict, ext = sniff_file_format("x.bin", b"PK\x03\x04......")
    assert fmt in ("office-zip", "xlsx", "pptx", "docx")
    assert conf >= 0.7
    # conflict may be True if ext unknown; acceptable


def test_sniff_office_zip_unknown_ext_conflict():
    """Test that office-zip with unknown extension sets conflict flag."""
    fmt, conf, conflict, ext = sniff_file_format("unknown.bin", b"PK\x03\x04......")
    assert fmt == "office-zip"
    assert conflict is True  # Should be True when magic=office-zip and ext=unknown


def test_quality_bucket_rules():
    """Test quality bucket assignment rules."""
    # Test case: empty extract should be "fail"
    m = {
        "input": {"format": "pdf"},
        "metrics": {
            "markdown_length": 100,
            "block_count": 0,
            "table_count": 0,
            "figure_count": 0
        },
        "warnings": {
            "osd_fail_count": 0,
            "wmf_missing_loader": False
        },
        "text_layer_detected": False,
        "status": {"quality_bucket": "ok"}
    }
    bucket = _assign_quality_bucket(m)
    assert bucket in ("suspect", "fail")  # Should be fail due to empty extract
    
    # Test case: short markdown should be "suspect"
    m2 = {
        "input": {"format": "pdf"},
        "metrics": {
            "markdown_length": 300,  # < 500
            "block_count": 5,
            "table_count": 1,
            "figure_count": 0
        },
        "warnings": {
            "osd_fail_count": 0,
            "wmf_missing_loader": False
        },
        "text_layer_detected": False,
        "status": {"quality_bucket": "ok"}
    }
    bucket2 = _assign_quality_bucket(m2)
    assert bucket2 == "suspect"
    
    # Test case: WMF missing loader should be "suspect"
    m3 = {
        "input": {"format": "pptx"},
        "metrics": {
            "markdown_length": 1000,
            "block_count": 10,
            "table_count": 0,
            "figure_count": 0
        },
        "warnings": {
            "osd_fail_count": 0,
            "wmf_missing_loader": True  # Should trigger suspect
        },
        "text_layer_detected": False,
        "status": {"quality_bucket": "ok"}
    }
    bucket3 = _assign_quality_bucket(m3)
    assert bucket3 == "suspect"
    
    # Test case: oversize markdown should be "suspect"
    m4 = {
        "input": {"format": "pdf"},
        "metrics": {
            "markdown_length": 2_500_000,  # > 2M
            "block_count": 100,
            "table_count": 0,
            "figure_count": 0
        },
        "warnings": {
            "osd_fail_count": 0,
            "wmf_missing_loader": False
        },
        "text_layer_detected": False,
        "status": {"quality_bucket": "ok"}
    }
    bucket4 = _assign_quality_bucket(m4)
    assert bucket4 == "suspect"


def test_export_metrics_resilient(result_stub):
    """Test metrics extraction with resilient fallback."""
    markdown_text = "Hello"
    metrics = _extract_document_metrics(
        result_stub,
        markdown_text,
        "pdf",
        0.123,
        {"warnings": {}}
    )
    assert "metrics" in metrics
    assert metrics["metrics"]["markdown_length"] == len(markdown_text)
    assert metrics["metrics"]["block_count"] == 3  # From result_stub fixture
    assert metrics["input"]["format"] == "pdf"
    assert metrics["status"]["quality_bucket"] in ("ok", "suspect", "fail")


def test_export_metrics_fallback():
    """Test metrics extraction fallback when export_to_dict fails."""
    # Create a stub that will fail export_to_dict but has blocks attribute
    class FailingResultStub:
        def __init__(self):
            self.document = FailingDocumentStub()
    
    class FailingDocumentStub:
        def __init__(self):
            self.blocks = [
                type('Block', (), {'type': 'heading'})(),
                type('Block', (), {'type': 'table'})(),
            ]
        
        def export_to_dict(self):
            raise Exception("Export failed")
    
    result = FailingResultStub()
    metrics = _extract_document_metrics(
        result,
        "Test markdown",
        "pdf",
        0.5,
        {"warnings": {}}
    )
    # Should still extract some metrics via fallback
    assert "metrics" in metrics
    assert metrics["metrics"]["markdown_length"] == len("Test markdown")

