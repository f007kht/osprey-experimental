"""Text processing utilities for normalization, chunking, and download preparation.

Module 1 - Osprey Backend: Document processing application.
"""

import hashlib
import json
import re
from typing import List, Dict, Any, Optional, Tuple


def content_hash(b: bytes) -> str:
    """Compute SHA256 hash of file content for idempotency.
    
    Args:
        b: File bytes to hash
        
    Returns:
        SHA256 hex digest
    """
    h = hashlib.sha256()
    h.update(b or b"")
    return h.hexdigest()


def normalize_text(
    text: str,
    fix_spacing: bool = True,
    fix_ligatures: bool = True,
    filter_ocr_artifacts: bool = True
) -> str:
    """
    Normalize text by fixing common OCR and formatting issues.

    Args:
        text: Input text to normalize
        fix_spacing: Fix spacing issues (multiple spaces, line breaks)
        fix_ligatures: Replace common ligatures with standard characters
        filter_ocr_artifacts: Remove OCR artifacts and junk characters

    Returns:
        Normalized text
    """
    if not text:
        return text

    # Fix ligatures (common in PDFs)
    if fix_ligatures:
        ligature_map = {
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            'ﬀ': 'ff',
            'ﬃ': 'ffi',
            'ﬄ': 'ffl',
            'ﬆ': 'st',
            'Ĳ': 'IJ',
            'ĳ': 'ij',
            'Œ': 'OE',
            'œ': 'oe',
            'Æ': 'AE',
            'æ': 'ae',
        }
        for ligature, replacement in ligature_map.items():
            text = text.replace(ligature, replacement)

    # Fix spacing issues
    if fix_spacing:
        # Normalize multiple spaces to single space
        text = re.sub(r' +', ' ', text)
        # Normalize multiple newlines (keep max 2)
        text = re.sub(r'\n\n\n+', '\n\n', text)
        # Remove trailing whitespace from each line
        text = re.sub(r' +\n', '\n', text)
        # Remove leading/trailing whitespace
        text = text.strip()

    # Filter OCR artifacts
    if filter_ocr_artifacts:
        # Remove standalone punctuation artifacts (common in figure/table OCR)
        # Pattern: lines with only punctuation, numbers, or very short junk
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are only punctuation/symbols (likely OCR noise)
            if stripped and not re.match(r'^[.,;:!?\-_=\+\*#@\$%\^&\(\)\[\]\{\}]+$', stripped):
                # Skip lines with excessive special characters (>50% of line)
                if len(stripped) > 2:
                    special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', stripped))
                    if special_chars / len(stripped) < 0.5:
                        cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)

    return text


def get_download_filename(base_filename: str, extension: str) -> str:
    """Generate a safe download filename with the specified extension.
    
    Args:
        base_filename: Base filename (may include path)
        extension: File extension (without leading dot)
        
    Returns:
        Safe filename with extension
    """
    # Extract just the filename without path
    base = base_filename.split('/')[-1].split('\\')[-1]
    # Remove extension if present
    if '.' in base:
        base = '.'.join(base.split('.')[:-1])
    # Sanitize filename (remove invalid characters)
    base = re.sub(r'[<>:"/\\|?*]', '_', base)
    return f"{base}.{extension}"


def prepare_download_data(
    result,
    format_type: str,
    base_filename: str = "document",
    normalize_text_flag: bool = False,
    fix_spacing: bool = True,
    fix_ligatures: bool = True,
    filter_ocr_artifacts: bool = False
) -> Tuple[str, str]:
    """
    Prepare data for download in the specified format.

    Args:
        result: ConversionResult from Docling
        format_type: One of 'markdown', 'json', 'txt', 'doctags', 'html'
        base_filename: Base filename to use for download (default: "document")
        normalize_text_flag: Apply text normalization (spacing, ligatures, OCR artifacts)
        fix_spacing: Fix spacing issues
        fix_ligatures: Replace ligatures
        filter_ocr_artifacts: Remove OCR artifacts

    Returns:
        tuple of (data_string, filename)
    """
    if format_type == "markdown":
        data = result.document.export_to_markdown()
        filename = get_download_filename(base_filename, "md")
    elif format_type == "html":
        # HTML export includes MathML for formulas if formula enrichment is enabled
        data = result.document.export_to_html()
        filename = get_download_filename(base_filename, "html")
    elif format_type == "json":
        data = json.dumps(result.document.export_to_dict(), indent=2)
        filename = get_download_filename(base_filename, "json")
    elif format_type == "txt":
        # Note: strict_text parameter is deprecated but still functional
        data = result.document.export_to_markdown(strict_text=True)
        filename = get_download_filename(base_filename, "txt")
    elif format_type == "doctags":
        # Use export_to_doctags() instead of deprecated export_to_document_tokens()
        data = result.document.export_to_doctags()
        filename = get_download_filename(base_filename, "doctags")
    else:
        raise ValueError(f"Unknown format type: {format_type}")

    # Apply text normalization if requested (not for JSON format)
    if normalize_text_flag and format_type != "json":
        data = normalize_text(
            data,
            fix_spacing=fix_spacing,
            fix_ligatures=fix_ligatures,
            filter_ocr_artifacts=filter_ocr_artifacts
        )

    return data, filename


def chunk_document(document) -> List[Dict[str, Any]]:
    """
    Chunk a DoclingDocument using HierarchicalChunker.
    
    Args:
        document: DoclingDocument instance
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    try:
        from docling_core.transforms.chunker import HierarchicalChunker
    except ImportError:
        raise ImportError("HierarchicalChunker not available. Install docling with required dependencies.")
    
    chunker = HierarchicalChunker()
    chunks = list(chunker.chunk(document))
    
    chunk_data = []
    for idx, chunk in enumerate(chunks):
        chunk_data.append({
            "text": chunk.text,
            "chunk_index": idx,
            "metadata": {
                "hierarchy_level": getattr(chunk, "hierarchy_level", None),
            }
        })
    
    return chunk_data


def handle_wmf_graphics(log_line: str, metrics: Dict[str, Any]) -> None:
    """
    Track WMF/EMF graphics that cannot be loaded.
    
    Args:
        log_line: Log line to check for WMF/EMF warnings
        metrics: Metrics dictionary to update
    """
    if 'WMF file' in log_line or 'EMF file' in log_line:
        metrics['rasterized_graphics_skipped'] = metrics.get('rasterized_graphics_skipped', 0) + 1
        metrics.setdefault('warnings', {}).update({'wmf_missing_loader': True})

