"""File format detection utilities.

Module 1 - Osprey Backend: Document processing application.
"""

from typing import Tuple, Optional


# Magic byte signatures for format detection
MAGIC = {
    b'%PDF-': 'pdf',
    b'PK\x03\x04': 'office-zip',  # xlsx/pptx/docx
}


def ext_to_office(fmt_ext: str) -> Optional[str]:
    """Extract Office format from file extension.
    
    Args:
        fmt_ext: File extension (with or without leading dot)
        
    Returns:
        Office format string ('xlsx', 'pptx', 'docx', 'pdf') or None
    """
    ext = (fmt_ext or '').lower().strip('.')
    if ext in ('xlsx',):
        return 'xlsx'
    if ext in ('pptx',):
        return 'pptx'
    if ext in ('docx',):
        return 'docx'
    if ext in ('pdf',):
        return 'pdf'
    return None


def looks_like_office_zip(first_bytes: bytes) -> bool:
    """Check if bytes start with Office ZIP magic signature.
    
    Args:
        first_bytes: First few bytes of the file
        
    Returns:
        True if bytes start with Office ZIP signature
    """
    return first_bytes.startswith(b"PK\x03\x04")


def looks_like_pdf(first_bytes: bytes) -> bool:
    """Check if bytes start with PDF magic signature.
    
    Args:
        first_bytes: First few bytes of the file
        
    Returns:
        True if bytes start with PDF signature
    """
    return first_bytes.startswith(b"%PDF")


def sniff_file_format(filename: str, first_bytes: bytes) -> Tuple[str, float, bool, Optional[str]]:
    """
    Detect file format using magic bytes and extension.
    
    Args:
        filename: Original filename
        first_bytes: First few bytes of the file
        
    Returns:
        Tuple of (format, confidence, conflict, ext_guess)
        - format: Detected format string
        - confidence: Confidence score 0.0-1.0
        - conflict: True if magic bytes and extension conflict
        - ext_guess: Format guessed from extension
    """
    mb_guess = None
    for sig, fmt in MAGIC.items():
        if first_bytes.startswith(sig):
            mb_guess = fmt
            break
    
    ext_guess = ext_to_office(filename.split('.')[-1])
    
    # Resolve format
    if mb_guess == 'pdf':
        resolved, confidence = 'pdf', 0.95
    elif mb_guess == 'office-zip':
        resolved = ext_guess or 'office-zip'
        confidence = 0.9 if ext_guess else 0.7
    else:
        resolved = ext_guess or 'unknown'
        confidence = 0.5 if ext_guess else 0.1
    
    # Check for conflict
    conflict = (mb_guess is not None and ext_guess is not None and
                not ((mb_guess == 'pdf' and ext_guess == 'pdf') or
                     (mb_guess == 'office-zip' and ext_guess in ('xlsx', 'pptx', 'docx'))))
    
    # Special case: if magic-bytes says office-zip and extension is unknown, set conflict flag
    if mb_guess == 'office-zip' and ext_guess is None:
        conflict = True
    
    return resolved, confidence, conflict, ext_guess

