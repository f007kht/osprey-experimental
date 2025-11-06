"""Pytest configuration and fixtures for quality gates tests."""

import pytest
from typing import Dict, Any


class ResultStub:
    """Stub class to mimic minimal result.document/export_to_dict behavior."""
    
    def __init__(self, blocks: list = None):
        self.document = DocumentStub(blocks or [])
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export to dict format."""
        return {"blocks": [b for b in self.document.blocks]}


class DocumentStub:
    """Stub class to mimic document.blocks behavior."""
    
    def __init__(self, blocks: list = None):
        self.blocks = blocks or []
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export to dict format."""
        return {"blocks": [b if isinstance(b, dict) else {"type": getattr(b, "type", "text")} for b in self.blocks]}


@pytest.fixture
def result_stub():
    """Fixture providing a minimal result stub."""
    return ResultStub(blocks=[
        {"type": "heading", "text": "Test Heading"},
        {"type": "paragraph", "text": "Test paragraph"},
        {"type": "table", "text": "Test table"},
    ])

