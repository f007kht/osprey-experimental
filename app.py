# -*- coding: utf-8 -*-
# Streamlit Document Processor App
#
# What this app does:
# - Provides a web-based interface for uploading and processing documents with Docling
# - Displays the extracted Markdown content in an interactive Streamlit app
#
# Requirements
# - Python 3.9+
# - Install Docling: `pip install docling`
# - Install Streamlit: `pip install streamlit`
#
# How to run
# - Run from the project root: `streamlit run app.py`
# - The app will open in your web browser at http://localhost:8501
#
# Notes
# - The converter auto-detects supported formats (PDF, DOCX, HTML, PPTX, images, etc.).
# - First run will download models, which may take several minutes.
# - For batch processing or saving outputs to files, see `docs/examples/batch_convert.py`.

import os
import json
import logging

# CRITICAL: Health check must be early (before heavy imports)
# This keeps health endpoint fast and cheap
import streamlit as st

# Handle health check query parameter early (before loading heavy dependencies)
try:
    # Try new query_params API first (Streamlit >= 1.28)
    params = st.query_params if hasattr(st, "query_params") else {}
    health_param = params.get("health", ["0"]) if isinstance(params, dict) else params.get("health", "0")
except Exception:
    # Fallback for older Streamlit versions
    try:
        params = st.experimental_get_query_params()
        health_param = params.get("health", ["0"])[0] if isinstance(params, dict) else "0"
    except Exception:
        health_param = "0"

if health_param in ("1", "true", "True"):
    # Minimal health check - return immediately without loading heavy dependencies
    from app_settings import get_settings
    
    settings = get_settings()
    payload = {
        "status": "ok",
        "mongo": {
            "enabled": settings.enable_mongodb,
            "status": getattr(settings, "mongo_status", "disabled")
        },
        "ocr": {
            "tessdata_prefix": os.environ.get("TESSDATA_PREFIX", "not_set")
        },
        "port": os.environ.get("PORT", "8501"),
        "version": os.environ.get("APP_VERSION", "dev"),
    }
    st.write(json.dumps(payload))
    st.stop()

# Now import heavy dependencies (after health check)
import tempfile
import signal
import time
import warnings
import io
import hashlib
import uuid
import re
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime

# CRITICAL: Remove DOCLING_ARTIFACTS_PATH BEFORE importing docling modules
# The settings singleton in docling/datamodel/settings.py caches environment variables
# at import time. If we don't remove it here, it will be cached even if we delete it later.
os.environ.pop("DOCLING_ARTIFACTS_PATH", None)

# Load application settings (must be done early, before other imports that depend on env vars)
from app_settings import get_settings
app_config = get_settings()

# Set environment variables from settings (for compatibility with existing code)
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = app_config.opencv_io_enable_openexr
os.environ["PYTHONIOENCODING"] = app_config.python_io_encoding
os.environ["PYTHONUTF8"] = app_config.python_utf8
os.environ["LC_ALL"] = app_config.lc_all

# Auto-detect TESSDATA_PREFIX if not already set (with graceful degradation)
# This ensures Tesseract OCR can find language data files on all platforms
# The Dockerfile sets this for production, but we detect it at runtime as a fallback
OCR_AVAILABLE = False
OCR_STATUS = "unavailable"
if not app_config.tessdata_prefix:
    # Check common Tesseract installation paths on Linux/Ubuntu systems
    for tessdata_path in [
        "/usr/share/tesseract-ocr/5/tessdata/",
        "/usr/share/tesseract-ocr/4.00/tessdata/",
        "/usr/share/tesseract-ocr/tessdata/",
    ]:
        if os.path.exists(tessdata_path):
            os.environ["TESSDATA_PREFIX"] = tessdata_path
            app_config.tessdata_prefix = tessdata_path
            OCR_AVAILABLE = True
            OCR_STATUS = "available"
            break
    else:
        # No TESSDATA_PREFIX found - will use CLI OCR mode (degraded but functional)
        OCR_STATUS = "degraded"
        logging.warning(
            "TESSDATA_PREFIX not found. OCR will use CLI mode (auto-detects language data). "
            "For better performance, set TESSDATA_PREFIX environment variable."
        )
else:
    # TESSDATA_PREFIX was set via environment variable
    if os.path.exists(app_config.tessdata_prefix):
        OCR_AVAILABLE = True
        OCR_STATUS = "available"
    else:
        OCR_STATUS = "degraded"
        logging.warning(
            f"TESSDATA_PREFIX set to {app_config.tessdata_prefix} but path does not exist. "
            "Falling back to CLI OCR mode."
        )

# Feature flags from settings
ENABLE_DOWNLOADS = app_config.enable_downloads
ENABLE_MONGODB = app_config.enable_mongodb

# QA feature flags (read once at startup)
QA_FLAG_ENABLE_PDF_WARNING_SUPPRESS = os.getenv("QA_FLAG_ENABLE_PDF_WARNING_SUPPRESS", "1") == "1"
QA_FLAG_ENABLE_TEXT_LAYER_DETECT = os.getenv("QA_FLAG_ENABLE_TEXT_LAYER_DETECT", "1") == "1"
QA_FLAG_LOG_NORMALIZED_CODES = os.getenv("QA_FLAG_LOG_NORMALIZED_CODES", "1") == "1"
QA_SCHEMA_VERSION = int(os.getenv("QA_SCHEMA_VERSION", "2"))

# Guardrails (configurable via env, with safe defaults)
MAX_PAGES = int(os.getenv("QA_MAX_PAGES", "500"))
MAX_SECONDS = float(os.getenv("QA_MAX_SECONDS", "300"))

# Secret scrubbing regex
SECRET_URI_RE = re.compile(r"(mongodb\+srv://|mongodb://)([^:@/]+):([^@/]+)@")


def _scrub_secrets(msg: str) -> str:
    """Scrub secrets (MongoDB connection strings, etc.) from log messages."""
    if not msg:
        return msg
    return SECRET_URI_RE.sub(r"\1***:***@", msg)


def _content_hash(b: bytes) -> str:
    """Compute SHA256 hash of file content for idempotency."""
    h = hashlib.sha256()
    h.update(b or b"")
    return h.hexdigest()

import streamlit as st

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions, TesseractCliOcrOptions
    from docling.datamodel.settings import settings
    from docling.document_converter import DocumentConverter, PdfFormatOption
except Exception as e:
    st.error(f"Failed to import docling modules: {e}")
    st.exception(e)
    st.stop()

# MEMORY OPTIMIZATION: Configure global settings to reduce memory pressure
# Page batch size is configurable via PAGE_BATCH_SIZE env var (default: 1)
# Higher values = faster processing but more memory usage
settings.perf.page_batch_size = app_config.page_batch_size

# CRITICAL: Override artifacts_path in settings singleton
# This ensures the cached environment variable value is cleared
# Even though we removed DOCLING_ARTIFACTS_PATH before import, we override as a safety measure
settings.artifacts_path = None

# Optional MongoDB and embedding imports
try:
    from pymongo import MongoClient
    from pymongo.operations import SearchIndexModel
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

try:
    from docling_core.transforms.chunker import HierarchicalChunker
    CHUNKER_AVAILABLE = True
except ImportError:
    CHUNKER_AVAILABLE = False

# Local embeddings (sentence-transformers)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Remote embeddings (VoyageAI - optional)
try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False

# Embedding configuration from settings
DEFAULT_EMBEDDING_MODEL = app_config.embedding_model

# MongoDB connection validation at startup (non-blocking, fast)
MONGODB_STATUS = "disabled"
MONGODB_DB_COLLECTION = None

if ENABLE_MONGODB and PYMONGO_AVAILABLE:
    is_valid, error_msg = app_config.validate_mongodb_config()
    if is_valid and app_config.mongodb_connection_string:
        # Use safe_mongo_ping for fast, non-blocking validation
        connected, status_msg = app_config.safe_mongo_ping(
            app_config.mongodb_connection_string,
            timeout_ms=2500
        )
        if connected:
            MONGODB_STATUS = "connected"
            MONGODB_DB_COLLECTION = f"{app_config.mongodb_database}/{app_config.mongodb_collection}"
            app_config.mongo_status = f"connected:{status_msg}"
            logging.info(f"MongoDB: connected to {MONGODB_DB_COLLECTION}")
        else:
            MONGODB_STATUS = "error"
            app_config.mongo_status = f"error:{status_msg}"
            logging.warning(f"MongoDB: connection failed - {status_msg}")
    elif error_msg:
        MONGODB_STATUS = "error"
        app_config.mongo_status = f"error:{error_msg}"
        logging.warning(f"MongoDB: configuration error - {error_msg}")
else:
    if ENABLE_MONGODB and not PYMONGO_AVAILABLE:
        MONGODB_STATUS = "error"
        app_config.mongo_status = "error:pymongo_not_available"
        logging.warning("MongoDB: enabled but pymongo not available")
    else:
        app_config.mongo_status = "disabled"
        logging.info("MongoDB: disabled (set ENABLE_MONGODB=true to enable)")

# Cache the converter to avoid reinitializing on every upload
# This prevents memory spikes from loading models multiple times
@st.cache_resource
def get_converter(
    enable_formula_enrichment: bool = False,
    enable_table_structure: bool = True,
    enable_code_enrichment: bool = False,
    enable_picture_classification: bool = False
):
    """
    Get or create a cached DocumentConverter instance with configurable options.

    Memory optimization: Image generation is ALWAYS enabled with optimized settings
    to reduce memory pressure while maintaining image functionality.

    Args:
        enable_formula_enrichment: Extract LaTeX representation of formulas
        enable_table_structure: Enable enhanced table structure extraction
        enable_code_enrichment: Enable code block language detection
        enable_picture_classification: Classify picture types (charts, diagrams, etc.)

    Returns:
        DocumentConverter instance configured with specified options
    """
    # Note: Cannot use Streamlit UI elements (st.error, st.exception) inside cached functions
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True

    # Choose Tesseract OCR engine based on environment
    # If TESSDATA_PREFIX is set and available, use the library version (faster)
    # Otherwise, use the CLI version (auto-detects language data, graceful degradation)
    if OCR_AVAILABLE and app_config.tessdata_prefix:
        pipeline_options.ocr_options = TesseractOcrOptions()
    else:
        pipeline_options.ocr_options = TesseractCliOcrOptions()

    # Force CPU device to avoid GPU/CUDA/MPS detection issues that can prevent Streamlit from starting
    # This ensures the app works on systems without GPU or with GPU driver issues
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU

    # Enable enrichment features
    pipeline_options.do_formula_enrichment = enable_formula_enrichment
    pipeline_options.do_table_structure = enable_table_structure
    pipeline_options.do_code_enrichment = enable_code_enrichment
    pipeline_options.do_picture_classification = enable_picture_classification

    # MEMORY OPTIMIZATION: Image generation with configurable settings
    # Settings can be tuned via IMAGES_ENABLE, IMAGES_SCALE, and PIPELINE_QUEUE_MAX env vars
    pipeline_options.generate_page_images = app_config.images_enable
    pipeline_options.generate_picture_images = app_config.images_enable
    pipeline_options.images_scale = app_config.images_scale  # Default 0.75 for stability with heavy PDFs

    # CRITICAL: Reduce queue sizes to prevent memory buildup
    # Default is 100 pages in queue, which can hold many high-res images in memory
    pipeline_options.queue_max_size = app_config.pipeline_queue_max  # Default 6 for stability

    # Reduce batch sizes to process one page at a time
    # This reduces peak memory usage during processing
    pipeline_options.ocr_batch_size = 1
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


@st.cache_resource
def get_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Get or create a cached embedding model instance."""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
    return SentenceTransformer(model_name)


def _get_download_filename(base_filename: str, extension: str) -> str:
    """Generate download filename from original filename."""
    name_without_ext = os.path.splitext(base_filename)[0]
    return f"{name_without_ext}.{extension}"


def _normalize_text(text: str, fix_spacing: bool = True, fix_ligatures: bool = True, filter_ocr_artifacts: bool = True) -> str:
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
    import re

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


def _prepare_download_data(
    result,
    format_type: str,
    base_filename: str = "document",
    normalize_text: bool = False,
    fix_spacing: bool = True,
    fix_ligatures: bool = True,
    filter_ocr_artifacts: bool = False
) -> tuple[str, str]:
    """
    Prepare data for download in the specified format.

    Args:
        result: ConversionResult from Docling
        format_type: One of 'markdown', 'json', 'txt', 'doctags', 'html'
        base_filename: Base filename to use for download (default: "document")
        normalize_text: Apply text normalization (spacing, ligatures, OCR artifacts)
        fix_spacing: Fix spacing issues
        fix_ligatures: Replace ligatures
        filter_ocr_artifacts: Remove OCR artifacts

    Returns:
        tuple of (data_string, filename)
    """
    if format_type == "markdown":
        data = result.document.export_to_markdown()
        filename = _get_download_filename(base_filename, "md")
    elif format_type == "html":
        # HTML export includes MathML for formulas if formula enrichment is enabled
        data = result.document.export_to_html()
        filename = _get_download_filename(base_filename, "html")
    elif format_type == "json":
        data = json.dumps(result.document.export_to_dict(), indent=2)
        filename = _get_download_filename(base_filename, "json")
    elif format_type == "txt":
        # Note: strict_text parameter is deprecated but still functional
        data = result.document.export_to_markdown(strict_text=True)
        filename = _get_download_filename(base_filename, "txt")
    elif format_type == "doctags":
        # Use export_to_doctags() instead of deprecated export_to_document_tokens()
        data = result.document.export_to_doctags()
        filename = _get_download_filename(base_filename, "doctags")
    else:
        raise ValueError(f"Unknown format type: {format_type}")

    # Apply text normalization if requested (not for JSON format)
    if normalize_text and format_type != "json":
        data = _normalize_text(
            data,
            fix_spacing=fix_spacing,
            fix_ligatures=fix_ligatures,
            filter_ocr_artifacts=filter_ocr_artifacts
        )

    return data, filename


def _chunk_document(document) -> List[Dict[str, Any]]:
    """
    Chunk a DoclingDocument using HierarchicalChunker.
    
    Args:
        document: DoclingDocument instance
    
    Returns:
        List of chunk dictionaries with text and metadata
    """
    if not CHUNKER_AVAILABLE:
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


def _generate_embeddings(chunk_texts: List[str], embedding_config: Optional[Dict[str, Any]] = None) -> List[List[float]]:
    """
    Generate embeddings for chunks using local or remote models.
    
    Args:
        chunk_texts: List of chunk text strings
        embedding_config: Optional dict with embedding configuration
            - use_remote: bool (default: False for local)
            - model_name: str (default: "sentence-transformers/all-MiniLM-L6-v2" for local)
            - api_key: str (required if use_remote=True)
    
    Returns:
        List of embedding vectors
    """
    if embedding_config is None:
        embedding_config = {"use_remote": False, "model_name": DEFAULT_EMBEDDING_MODEL}
    
    use_remote = embedding_config.get("use_remote", False)
    model_name = embedding_config.get("model_name", DEFAULT_EMBEDDING_MODEL)
    
    if use_remote:
        # Use VoyageAI (remote)
        if not VOYAGEAI_AVAILABLE:
            raise ImportError("voyageai not available. Install with: pip install voyageai")
        api_key = embedding_config.get("api_key")
        if not api_key:
            raise ValueError("VoyageAI API key is required for remote embeddings")
        
        vo = voyageai.Client(api_key)
        result = vo.contextualized_embed(
            inputs=[chunk_texts],
            model="voyage-context-3"
        )
        embeddings = [emb for r in result.results for emb in r.embeddings]
        return embeddings
    else:
        # Use local sentence-transformers
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not available. Install with: pip install sentence-transformers")
        
        model = get_embedding_model(model_name)
        embeddings = model.encode(chunk_texts, show_progress_bar=False)
        return embeddings.tolist()  # Convert numpy array to list


# Magic byte signatures for format detection
MAGIC = {
    b'%PDF-': 'pdf',
    b'PK\x03\x04': 'office-zip',  # xlsx/pptx/docx
}


def _ext_to_office(fmt_ext: str) -> Optional[str]:
    """Extract Office format from file extension."""
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


def sniff_file_format(filename: str, first_bytes: bytes) -> Tuple[str, float, bool, Optional[str]]:
    """
    Detect file format using magic bytes and extension.
    
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
    
    ext_guess = _ext_to_office(filename.split('.')[-1])
    
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


class _PdfNoiseFilter(logging.Filter):
    """Filter to block PDF noise messages (PK/EOF warnings) from non-PDF flows."""
    BLOCK_PATTERNS = (
        "invalid pdf header: b'PK\\x03\\x04\\x14'",
        "EOF marker not found",
    )
    
    def filter(self, record):
        msg = str(record.getMessage() or "")
        return not any(p in msg for p in self.BLOCK_PATTERNS)


_pdf_noise_filter = _PdfNoiseFilter()


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


def _handle_wmf_graphics(log_line: str, metrics: dict):
    """
    Track WMF/EMF graphics that cannot be loaded.
    
    Args:
        log_line: Log line to check for WMF/EMF warnings
        metrics: Metrics dictionary to update
    """
    if 'WMF file' in log_line or 'EMF file' in log_line:
        metrics['rasterized_graphics_skipped'] = metrics.get('rasterized_graphics_skipped', 0) + 1
        metrics.setdefault('warnings', {}).update({'wmf_missing_loader': True})


def _detect_pdf_text_layer(pdf_bytes: bytes, max_pages_check: int = 3) -> Tuple[bool, list]:
    """
    Detect if PDF has embedded text layer by checking first few pages.
    
    Args:
        pdf_bytes: PDF file content as bytes
        max_pages_check: Maximum number of pages to check
        
    Returns:
        Tuple of (has_text_layer, per_page_flags)
        - has_text_layer: True if any checked page has text
        - per_page_flags: List of booleans for each checked page
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        flags = []
        for i, page in enumerate(reader.pages[:max_pages_check]):
            txt = page.extract_text()
            # Treat None or whitespace as "no text"; require >=15 visible chars
            if txt is None or not txt.strip():
                flags.append(False)
            else:
                flags.append(len(txt.strip()) >= 15)
        return (any(flags), flags)
    except Exception:
        return (False, [])


def _should_disable_ocr_for_page1(osd_fail_count: int) -> bool:
    """
    Determine if OCR should be disabled for page 1 after repeated OSD failures.
    
    Args:
        osd_fail_count: Number of OSD failures encountered
        
    Returns:
        True if OCR should be disabled
    """
    return osd_fail_count >= 3


def _normalize_ocr_engine_name(engine) -> str:
    """
    Normalize OCR engine name to a stable string identifier.
    
    Args:
        engine: OCR engine object or name
        
    Returns:
        Normalized engine name string
    """
    if not engine:
        return 'none'
    name = str(engine).lower()
    for k in ('tesseract', 'rapidocr', 'easyocr', 'auto'):
        if k in name:
            return k
    return name[:32]


def _extract_document_metrics(result, markdown_text: str, fmt: str, process_seconds: float, extras: dict) -> dict:
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
        d["status"]["quality_bucket"] = _assign_quality_bucket(d)
    return d


def _assign_quality_bucket(m: dict) -> str:
    """
    Assign quality bucket (ok/suspect/fail) based on metrics.
    
    Args:
        m: Metrics dictionary from _extract_document_metrics
        
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


def _sanitize_for_mongodb(obj: Any) -> Any:
    """
    Recursively sanitize data for MongoDB BSON compatibility.

    MongoDB can only handle integers up to 64-bit (8 bytes):
    - Min: -9,223,372,036,854,775,808
    - Max: 9,223,372,036,854,775,807

    Python's int type can be arbitrarily large, so we need to convert
    any integers exceeding this range to strings to avoid OverflowError.

    Args:
        obj: Any Python object (dict, list, int, float, str, etc.)

    Returns:
        Sanitized object safe for MongoDB BSON encoding
    """
    # MongoDB's 64-bit integer limits
    MIN_BSON_INT = -9223372036854775808
    MAX_BSON_INT = 9223372036854775807

    if isinstance(obj, dict):
        # Recursively sanitize all dictionary values
        return {key: _sanitize_for_mongodb(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # Recursively sanitize all list items
        return [_sanitize_for_mongodb(item) for item in obj]
    elif isinstance(obj, tuple):
        # Convert tuples to lists and sanitize
        return [_sanitize_for_mongodb(item) for item in obj]
    elif isinstance(obj, int):
        # Check if integer is within BSON range
        if obj < MIN_BSON_INT or obj > MAX_BSON_INT:
            # Convert to string if out of range
            return str(obj)
        return obj
    elif isinstance(obj, (str, float, bool, type(None))):
        # These types are safe as-is
        return obj
    else:
        # For any other type, try to convert to string
        # This handles datetime, custom objects, etc.
        try:
            return str(obj)
        except Exception:
            # If conversion fails, return None
            return None


def _store_in_mongodb(
    document: Any,
    original_filename: str,
    file_size: int,
    mongodb_connection_string: str,
    database_name: str,
    collection_name: str,
    embedding_config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    max_retries: int = 3
) -> bool:
    """
    Store processed document in MongoDB with vector embeddings.
    
    Args:
        document: DoclingDocument instance
        original_filename: Original uploaded filename
        file_size: Size of original file in bytes
        mongodb_connection_string: MongoDB connection string
        database_name: Database name
        collection_name: Collection name
        embedding_config: Optional dict with embedding configuration
            - use_remote: bool (default: False for local)
            - model_name: str (default: "sentence-transformers/all-MiniLM-L6-v2")
            - api_key: str (required if use_remote=True)
    
    Returns:
        True if successful, False otherwise
    """
    if not PYMONGO_AVAILABLE:
        return False
    
    if embedding_config is None:
        embedding_config = {"use_remote": False, "model_name": DEFAULT_EMBEDDING_MODEL}
    
    try:
        # Connect to MongoDB with SSL/TLS configuration
        # MongoDB Atlas requires proper SSL/TLS setup
        import certifi
        from pymongo.errors import OperationFailure

        # Parse connection string to check format
        # MongoDB Atlas requires mongodb+srv:// format
        if not mongodb_connection_string.startswith('mongodb+srv://') and not mongodb_connection_string.startswith('mongodb://'):
            raise ValueError("Invalid MongoDB connection string format. Use mongodb+srv://...")

        # Mask connection string for logging
        masked_uri = app_config.mask_connection_string(mongodb_connection_string)
        logging.info(f"MongoDB: connecting with URI {masked_uri}")

        # Configure MongoClient with proper SSL/TLS settings
        # Use pymongo-compatible parameter names (tls, tlsCAFile - NOT ssl_context)
        client = MongoClient(
            mongodb_connection_string,
            tls=True,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=False,
            tlsAllowInvalidHostnames=False,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=20000,
            socketTimeoutMS=20000,
            retryWrites=True,
            retryReads=True,
            directConnection=False
        )

        # Test the connection before proceeding
        try:
            client.admin.command('ping')
            logging.info("MongoDB: connected successfully")
        except OperationFailure as auth_error:
            if auth_error.code == 8000:
                # Authentication failed - likely password changed
                logging.error(
                    f"MongoDB auth failed (code 8000) with URI {masked_uri}. "
                    "Did the password change? If it contains special characters (@:/#?&% etc.), "
                    "URL-encode it using urllib.parse.quote_plus()."
                )
            raise

        db = client[database_name]
        collection = db[collection_name]
        
        # Chunk the document
        chunks_data = _chunk_document(document)
        chunk_texts = [chunk["text"] for chunk in chunks_data]
        
        # Generate embeddings
        embeddings = _generate_embeddings(chunk_texts, embedding_config)
        
        # Determine embedding dimensions and model name
        embedding_dim = len(embeddings[0]) if embeddings else 0
        model_name = embedding_config.get("model_name", DEFAULT_EMBEDDING_MODEL)
        
        # Combine chunks with embeddings
        chunks_with_embeddings = []
        for chunk_data, embedding in zip(chunks_data, embeddings):
            chunks_with_embeddings.append({
                "text": chunk_data["text"],
                "embedding": embedding,
                "chunk_index": chunk_data["chunk_index"],
                "metadata": chunk_data["metadata"]
            })
        
        # Import MongoDB size helpers
        from helpers_mongo import split_for_mongo, bson_len, MAX_BSON_SAFE
        
        # Prepare primary document for storage
        primary_doc = {
            "filename": original_filename,
            "original_filename": original_filename,
            "processed_at": datetime.utcnow().isoformat(),
            "file_size": file_size,
            "docling_json": document.export_to_dict(),
            "markdown": document.export_to_markdown(),
            "chunks": chunks_with_embeddings,
            "metadata": {
                "total_chunks": len(chunks_with_embeddings),
                "embedding_model": model_name,
                "embedding_dimensions": embedding_dim,
                "use_remote": embedding_config.get("use_remote", False)
            }
        }
        
        # Add quality metrics if provided
        if metrics:
            primary_doc.update({
                "input": metrics.get("input", {}),
                "metrics": metrics.get("metrics", {}),
                "warnings": metrics.get("warnings", {}),
                "ocr": metrics.get("ocr", {}),
                "status": metrics.get("status", {}),
                "text_layer_detected": metrics.get("text_layer_detected", False),
                "rasterized_graphics_skipped": metrics.get("rasterized_graphics_skipped", 0),
                "schema_version": QA_SCHEMA_VERSION,
            })
        
        # Add correlation IDs for log <-> Mongo joinability (from metrics dict if provided)
        if metrics and isinstance(metrics, dict):
            run_id = metrics.get("run_id")
            content_hash = metrics.get("content_hash")
            if run_id:
                primary_doc["run_id"] = run_id
            if content_hash:
                primary_doc["content_hash"] = content_hash

        # Sanitize document for MongoDB BSON compatibility
        primary_doc = _sanitize_for_mongodb(primary_doc)
        
        # Helper function for exponential backoff retry
        import time
        def retry_with_backoff(operation, operation_name, max_retries=max_retries):
            """Retry operation with exponential backoff."""
            for attempt in range(max_retries):
                try:
                    return operation()
                except Exception as e:
                    if attempt == max_retries - 1:
                        # Last attempt, re-raise
                        raise
                    # Exponential backoff: 1s, 2s, 4s
                    wait_time = 2 ** attempt
                    logging.warning(
                        f"MongoDB {operation_name} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
        
        # Check if we need to split due to size
        # If document is large, split into primary + pages
        if bson_len(primary_doc) >= MAX_BSON_SAFE:
            # Extract pages from docling_json for separate storage
            docling_dict = primary_doc.get("docling_json", {})
            pages_raw = docling_dict.get("pages", [])
            
            # Convert pages to list of dicts for splitting
            page_docs_raw = []
            for i, page in enumerate(pages_raw):
                if isinstance(page, dict):
                    page_docs_raw.append(page)
                else:
                    # If page is not a dict, convert it
                    page_docs_raw.append({"page_index": i, "content": str(page)})
            
            # Split primary doc from pages
            small_primary, page_docs = split_for_mongo(primary_doc, page_docs_raw)
            
            # Idempotent upsert by content_hash + filename (if available)
            content_hash = small_primary.get("content_hash")
            filename = small_primary.get("filename")
            if content_hash and filename:
                primary_result = retry_with_backoff(
                    lambda: collection.update_one(
                        {"content_hash": content_hash, "filename": filename},
                        {"$set": small_primary},
                        upsert=True
                    ),
                    "primary document upsert"
                )
                primary_id = primary_result.upserted_id if primary_result.upserted_id else collection.find_one(
                    {"content_hash": content_hash, "filename": filename}
                )["_id"]
            else:
                # Fallback to insert if content_hash missing
                primary_result = retry_with_backoff(
                    lambda: collection.insert_one(small_primary),
                    "primary document insert"
                )
                primary_id = primary_result.inserted_id
            
            # Insert pages into separate collection (pages collection) with retry
            pages_collection = db[f"{collection_name}_pages"]
            for i, page_doc in enumerate(page_docs):
                page_doc["parent_id"] = primary_id
                page_doc["page_index"] = i
                # Use default parameter to capture page_doc in closure correctly
                def insert_page(pd=page_doc):
                    return pages_collection.insert_one(pd)
                retry_with_backoff(insert_page, f"page {i} insert")
            
            result = primary_result
            logging.info(f"Document split: primary ({bson_len(small_primary)} bytes) + {len(page_docs)} pages")
        else:
            # Document fits in single BSON doc - idempotent upsert by content_hash + filename
            content_hash = primary_doc.get("content_hash")
            filename = primary_doc.get("filename")
            if content_hash and filename:
                result = retry_with_backoff(
                    lambda: collection.update_one(
                        {"content_hash": content_hash, "filename": filename},
                        {"$set": primary_doc},
                        upsert=True
                    ),
                    "document upsert"
                )
            else:
                # Fallback to insert if content_hash missing
                result = retry_with_backoff(
                    lambda: collection.insert_one(primary_doc),
                    "document insert"
                )
        
        # Create vector search index if it doesn't exist
        # Note: Index creation is asynchronous and may take time
        # This is best done manually in MongoDB Atlas UI or via separate script
        # But we attempt it here for convenience
        try:
            search_index_model = SearchIndexModel(
                definition={
                    "fields": [{
                        "type": "vector",
                        "path": "chunks.embedding",
                        "numDimensions": embedding_dim,  # Dynamic based on model
                        "similarity": "dotProduct"
                    }]
                },
                name="vector_index",
                type="vectorSearch"
            )
            # This may raise an error if index already exists, which is fine
            collection.create_search_index(model=search_index_model)
        except Exception as idx_error:
            # Index creation might fail if it already exists or lacks permissions
            # This is not critical - document is already stored
            # User can create index manually in MongoDB Atlas UI
            pass
        
        return True
    
    except Exception as e:
        # Log the error but return False to honor the function contract
        # The function signature promises -> bool, so we return False instead of raising
        # Callers can check the return value and handle errors appropriately
        from pymongo.errors import OperationFailure
        
        if isinstance(e, OperationFailure) and e.code == 8000:
            masked_uri = app_config.mask_connection_string(mongodb_connection_string)
            logging.error(
                f"MongoDB auth failed (code 8000) with URI {masked_uri}. "
                "Did the password change? If it contains special characters (@:/#?&% etc.), "
                "URL-encode it using urllib.parse.quote_plus()."
            )
        else:
            logging.error(f"Error storing document in MongoDB: {e}", exc_info=True)
        return False
    finally:
        if 'client' in locals():
            client.close()

# Initialize session state for MongoDB configuration
if "mongodb_enabled" not in st.session_state:
    st.session_state.mongodb_enabled = False
if "mongodb_connection_string" not in st.session_state:
    st.session_state.mongodb_connection_string = ""
if "mongodb_database" not in st.session_state:
    st.session_state.mongodb_database = "docling_db"
if "mongodb_collection" not in st.session_state:
    st.session_state.mongodb_collection = "documents"
if "voyageai_api_key" not in st.session_state:
    st.session_state.voyageai_api_key = ""
if "use_remote_embeddings" not in st.session_state:
    st.session_state.use_remote_embeddings = False
if "embedding_model_name" not in st.session_state:
    st.session_state.embedding_model_name = DEFAULT_EMBEDDING_MODEL

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # MongoDB Configuration (only if enabled)
    if ENABLE_MONGODB:
        st.subheader("MongoDB Storage")
        mongodb_enabled = st.checkbox(
            "Enable MongoDB Storage",
            value=True,
            help="Store processed documents in MongoDB with vector embeddings for RAG"
        )
        
        mongodb_connection_string = st.text_input(
            "MongoDB Connection String",
            value=app_config.mongodb_connection_string or "",
            type="password",
            help="MongoDB Atlas connection string (e.g., mongodb+srv://...)",
            disabled=not mongodb_enabled
        )
        
        mongodb_database = st.text_input(
            "Database Name",
            value=app_config.mongodb_database,
            help="Name of the MongoDB database",
            disabled=not mongodb_enabled
        )
        
        mongodb_collection = st.text_input(
            "Collection Name",
            value=app_config.mongodb_collection,
            help="Name of the MongoDB collection",
            disabled=not mongodb_enabled
        )
        
        st.divider()
        st.subheader("Embedding Configuration")
        
        use_remote_embeddings = st.checkbox(
            "Use Remote Embeddings (VoyageAI)",
            value=app_config.use_remote_embeddings,
            help="Use VoyageAI for embeddings (requires API key). Unchecked = local models.",
            disabled=not mongodb_enabled
        )
        
        if use_remote_embeddings:
            voyageai_api_key = st.text_input(
                "VoyageAI API Key",
                value=app_config.voyageai_api_key or "",
                type="password",
                help="API key for VoyageAI embeddings (required if using remote)",
                disabled=not mongodb_enabled
            )
            embedding_model_name = "voyage-context-3"  # Fixed for VoyageAI
        else:
            voyageai_api_key = ""
            embedding_model_name = st.text_input(
                "Local Embedding Model",
                value=app_config.embedding_model,
                help="HuggingFace model name for local embeddings (e.g., sentence-transformers/all-MiniLM-L6-v2)",
                disabled=not mongodb_enabled
            )
        
        if mongodb_enabled and not mongodb_connection_string:
            st.warning("Warning: MongoDB connection string is required for storage.")
        if mongodb_enabled and use_remote_embeddings and not voyageai_api_key:
            st.warning("Warning: VoyageAI API key is required when using remote embeddings.")
    else:
        mongodb_enabled = False
        mongodb_connection_string = ""
        mongodb_database = "docling_db"
        mongodb_collection = "documents"
        voyageai_api_key = ""
        use_remote_embeddings = False
        embedding_model_name = DEFAULT_EMBEDDING_MODEL

    # Document Processing Options
    st.divider()
    st.subheader("Document Processing Options")

    enable_formula_enrichment = st.checkbox(
        "Extract Formulas (LaTeX)",
        value=False,
        help="Extract LaTeX representation of mathematical formulas. Increases processing time."
    )

    enable_table_structure = st.checkbox(
        "Enhanced Table Structure",
        value=True,
        help="Extract detailed table structure with cells, rows, and columns."
    )

    enable_code_enrichment = st.checkbox(
        "Code Language Detection",
        value=False,
        help="Detect programming language in code blocks. Increases processing time."
    )

    enable_picture_classification = st.checkbox(
        "Picture Classification",
        value=False,
        help="Classify picture types (charts, diagrams, logos, etc.). Increases processing time."
    )

    # Text Normalization Options
    st.divider()
    st.subheader("Text Normalization")

    apply_text_normalization = st.checkbox(
        "Apply Text Normalization",
        value=False,
        help="Fix spacing, ligatures, and OCR artifacts in output text."
    )

    if apply_text_normalization:
        fix_spacing = st.checkbox(
            "Fix Spacing Issues",
            value=True,
            help="Remove extra spaces and normalize line breaks."
        )

        fix_ligatures = st.checkbox(
            "Replace Ligatures",
            value=True,
            help="Replace ligatures (ﬁ, ﬂ, etc.) with standard characters (fi, fl)."
        )

        filter_ocr_artifacts = st.checkbox(
            "Filter OCR Artifacts",
            value=True,
            help="Remove junk characters and punctuation-only lines from OCR."
        )
    else:
        fix_spacing = True
        fix_ligatures = True
        filter_ocr_artifacts = False

    # Store in session state for use in processing
    st.session_state.mongodb_enabled = mongodb_enabled if ENABLE_MONGODB else False
    st.session_state.mongodb_connection_string = mongodb_connection_string
    st.session_state.mongodb_database = mongodb_database
    st.session_state.mongodb_collection = mongodb_collection
    st.session_state.voyageai_api_key = voyageai_api_key
    st.session_state.use_remote_embeddings = use_remote_embeddings if ENABLE_MONGODB else False
    st.session_state.embedding_model_name = embedding_model_name if ENABLE_MONGODB else DEFAULT_EMBEDDING_MODEL
    st.session_state.enable_formula_enrichment = enable_formula_enrichment
    st.session_state.enable_table_structure = enable_table_structure
    st.session_state.enable_code_enrichment = enable_code_enrichment
    st.session_state.enable_picture_classification = enable_picture_classification
    st.session_state.apply_text_normalization = apply_text_normalization
    st.session_state.fix_spacing = fix_spacing
    st.session_state.fix_ligatures = fix_ligatures
    st.session_state.filter_ocr_artifacts = filter_ocr_artifacts

# Set up the Streamlit page title
st.title("Docling Document Processor")
st.write("Upload a document (PDF, DOCX, PPTX, etc.) to process it with Docling.")

# 1. Create the document upload button
uploaded_file = st.file_uploader(
    "Choose a document...",
    type=["pdf", "docx", "pptx", "xlsx", "html", "png", "jpg", "jpeg", "tiff", "wav", "mp3"],
)

# 2. Run the docling process when a file is uploaded
if uploaded_file is not None:
    # Use a temporary file to store the uploaded content
    # Docling's .convert() method works well with file paths
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Read first 8 bytes for magic sniffing
    with open(tmp_file_path, 'rb') as f:
        first_bytes = f.read(8)
    
    # Sniff file format with new signature
    fmt, confidence, conflict, ext_guess = sniff_file_format(uploaded_file.name, first_bytes)
    
    # Generate correlation IDs for log <-> Mongo joinability
    run_id = str(uuid.uuid4())
    file_bytes = uploaded_file.getvalue()
    content_hash = _content_hash(file_bytes)
    
    # Initialize extras dict for tracking metrics
    extras = {
        "run_id": run_id,
        "content_hash": content_hash,
        "warnings": {},
        "rasterized_graphics_skipped": 0,
        "text_layer_detected": False,
        "ocr_engine": "unknown"
    }
    
        # Track format conflict
    if conflict:
        extras["warnings"]["format_conflict"] = True
        # Special logging for office-zip with unknown extension
        if fmt == "office-zip" and ext_guess is None:
            logging.warning(_scrub_secrets(f"WARNING - FORMAT_CONFLICT: magic=office-zip ext=unknown filename={uploaded_file.name} run_id={run_id} hash={content_hash[:8]}"))
        else:
            logging.warning(_scrub_secrets(f"Format conflict detected: magic bytes vs extension (FORMAT_CONFLICT) run_id={run_id} hash={content_hash[:8]}"))
    
    st.info(f"Processing `{uploaded_file.name}`... This might take a moment.")

    try:
        # Get the cached converter (initializes once, then reuses)
        # This prevents memory spikes from loading models multiple times
        # Pass processing options from session state
        with st.spinner("Loading Docling (this may take a while on first run)..."):
            converter = get_converter(
                enable_formula_enrichment=st.session_state.get("enable_formula_enrichment", False),
                enable_table_structure=st.session_state.get("enable_table_structure", True),
                enable_code_enrichment=st.session_state.get("enable_code_enrichment", False),
                enable_picture_classification=st.session_state.get("enable_picture_classification", False)
            )
        
        # For PDF files, detect text layer and configure OCR accordingly
        if fmt == "pdf" and QA_FLAG_ENABLE_TEXT_LAYER_DETECT:
            pdf_bytes = uploaded_file.getvalue()
            has_text_layer, page_flags = _detect_pdf_text_layer(pdf_bytes)
            if has_text_layer:
                extras["text_layer_detected"] = True
                extras["ocr_engine"] = "none"  # Audit trail: OCR disabled due to text layer
                logging.info("PDF text layer detected - OCR and OSD disabled for this document")
        
        # Track OSD failures and WMF warnings by intercepting log messages
        osd_fail_count = 0
        
        # Custom log handler to intercept WMF/OSD warnings
        # NOTE: Handler is attached per conversion and always detached in finally block
        class MetricsLogHandler(logging.Handler):
            def __init__(self, extras_dict):
                super().__init__()
                self.extras = extras_dict
                # Reset counters per run to avoid cross-doc leakage
                self.extras.setdefault("warnings", {})["osd_fail_count"] = 0
                
            def emit(self, record):
                msg = self.format(record)
                # Scrub secrets from log messages
                msg_scrubbed = _scrub_secrets(msg)
                
                # Track OSD failures with collapse logic for text-layer PDFs
                if 'OSD failed' in msg_scrubbed:
                    warnings_dict = self.extras.setdefault("warnings", {})
                    # Increment count
                    warnings_dict["osd_fail_count"] = warnings_dict.get("osd_fail_count", 0) + 1
                    
                    # Collapse spam: if text layer detected and this is page 0, suppress after first
                    if self.extras.get("text_layer_detected") and "page: 0" in msg_scrubbed:
                        if not self.extras.get("osd_suppressed_after_first"):
                            # First OSD failure on page 0 with text layer - mark for suppression
                            self.extras["osd_suppressed_after_first"] = True
                        # Swallow subsequent OSD messages if suppressed
                        if self.extras.get("osd_suppressed_after_first"):
                            # Filter out this record to prevent it from being emitted
                            return
                
                # Track WMF/EMF warnings
                if 'WMF' in msg_scrubbed or 'EMF' in msg_scrubbed or 'cannot be loaded by Pillow' in msg_scrubbed:
                    _handle_wmf_graphics(msg_scrubbed, self.extras)
                
                # Note: This handler is for tracking only; actual log emission happens via root logger
        
        # Add log handler for metrics tracking
        metrics_handler = MetricsLogHandler(extras)
        metrics_handler.setLevel(logging.WARNING)
        # Get docling loggers
        docling_logger = logging.getLogger('docling')
        docling_logger.addHandler(metrics_handler)
        
        # Add filter to suppress OSD messages when collapsed
        class OSDSuppressFilter(logging.Filter):
            def __init__(self, extras_dict):
                super().__init__()
                self.extras = extras_dict
            def filter(self, record):
                msg = str(record.getMessage() or "")
                # Suppress OSD messages if we've marked for suppression
                if 'OSD failed' in msg and self.extras.get("osd_suppressed_after_first"):
                    return False  # Filter out this record
                return True  # Allow other messages
        
        osd_filter = OSDSuppressFilter(extras)
        docling_logger.addFilter(osd_filter)
        
        try:
            # Run the conversion process
            with st.spinner(f"Converting `{uploaded_file.name}`..."):
                start_time = time.time()
                print(f"\n{'='*60}")
                print(f"CONVERSION START: {uploaded_file.name}")
                print(f"{'='*60}")

                # Determine if this is a non-PDF format (for noise suppression)
                is_non_pdf = fmt in ('xlsx', 'pptx', 'docx', 'office-zip')
                should_suppress = QA_FLAG_ENABLE_PDF_WARNING_SUPPRESS and is_non_pdf
                
                # Wrap entire conversion in PDF noise suppression for non-PDF files
                with suppress_pdf_noise_for_non_pdf(should_suppress):
                    # Option B: Detect document size and use chunked processing for large documents
                    # This prevents the 127-page stopping issue by processing in manageable chunks
                    try:
                        from pypdf import PdfReader
                        pdf_reader = PdfReader(tmp_file_path)
                        total_pages = len(pdf_reader.pages)
                        print(f"Document has {total_pages} pages")

                        # Guardrail: Check MAX_PAGES limit
                        if total_pages > MAX_PAGES:
                            extras["abort"] = {"reason": "MAX_PAGES", "limit": MAX_PAGES, "actual": total_pages}
                            logging.warning(_scrub_secrets(f"WARNING - MAX_PAGES exceeded: {total_pages} > {MAX_PAGES} run_id={run_id} hash={content_hash[:8]}"))
                            st.warning(f"⚠️ Document exceeds maximum page limit ({total_pages} > {MAX_PAGES} pages). Processing aborted.")
                            # Create a minimal result stub for metrics extraction
                            result = None
                        # For documents > 120 pages, process in chunks to avoid memory/timeout issues
                        elif total_pages > 120:
                            MAX_PAGES_PER_CHUNK = 120
                            st.warning(f"⚠️ Large document detected ({total_pages} pages). Processing in chunks of {MAX_PAGES_PER_CHUNK} pages for stability.")
                            print(f"Large document: processing first {MAX_PAGES_PER_CHUNK} pages of {total_pages}")

                            # Track conversion phases
                            print(f"Phase 1: Starting converter.convert() with page_range=(1, {MAX_PAGES_PER_CHUNK})...")
                            result = converter.convert(tmp_file_path, page_range=(1, MAX_PAGES_PER_CHUNK))
                            print(f"Phase 1 complete: converter.convert() returned in {time.time() - start_time:.2f}s")

                            st.info(f"✓ Processed first {MAX_PAGES_PER_CHUNK} pages. Additional chunks can be processed separately.")
                        else:
                            # Normal processing for documents <= 120 pages
                            print("Phase 1: Starting converter.convert()...")
                            result = converter.convert(tmp_file_path)
                            print(f"Phase 1 complete: converter.convert() returned in {time.time() - start_time:.2f}s")
                        
                        # Guardrail: Check MAX_SECONDS limit
                        elapsed = time.time() - start_time
                        if elapsed > MAX_SECONDS:
                            extras["abort"] = {"reason": "MAX_SECONDS", "limit": MAX_SECONDS, "actual": elapsed}
                            logging.warning(_scrub_secrets(f"WARNING - MAX_SECONDS exceeded: {elapsed:.2f}s > {MAX_SECONDS}s run_id={run_id} hash={content_hash[:8]}"))
                            st.warning(f"⚠️ Processing exceeded maximum time limit ({elapsed:.1f}s > {MAX_SECONDS}s). Further processing aborted.")
                            if 'result' in locals() and result is not None:
                                # Mark result as incomplete
                                pass

                    except Exception as e:
                        # Fallback to normal processing if page detection fails
                        print(f"Could not detect page count: {e}. Using normal processing.")
                        print("Phase 1: Starting converter.convert()...")
                        # Already wrapped in suppress_pdf_noise_for_non_pdf above
                        result = converter.convert(tmp_file_path)
                        print(f"Phase 1 complete: converter.convert() returned in {time.time() - start_time:.2f}s")
                        
                        # Guardrail: Check MAX_SECONDS limit after fallback
                        elapsed = time.time() - start_time
                        if elapsed > MAX_SECONDS:
                            extras["abort"] = {"reason": "MAX_SECONDS", "limit": MAX_SECONDS, "actual": elapsed}
                            logging.warning(_scrub_secrets(f"WARNING - MAX_SECONDS exceeded: {elapsed:.2f}s > {MAX_SECONDS}s run_id={run_id} hash={content_hash[:8]}"))
                            st.warning(f"⚠️ Processing exceeded maximum time limit ({elapsed:.1f}s > {MAX_SECONDS}s). Further processing aborted.")
        finally:
            # Always remove log handler and filter after conversion (even on error)
            try:
                docling_logger.removeHandler(metrics_handler)
                docling_logger.removeFilter(osd_filter)
            except Exception:
                pass
        
        # Count OSD failures from result errors (backup count)
        if 'result' in locals() and hasattr(result, 'errors') and result.errors:
            for error in result.errors:
                error_msg = str(error.error_message) if hasattr(error, 'error_message') else str(error)
                if 'OSD failed' in error_msg or 'osd' in error_msg.lower():
                    osd_fail_count += 1
                # Track WMF/EMF warnings from errors (backup)
                if 'WMF' in error_msg or 'EMF' in error_msg or 'cannot be loaded by Pillow' in error_msg:
                    _handle_wmf_graphics(error_msg, extras)
        
        # Use the count from log handler (which may be more accurate)
        # Fall back to error count if log handler didn't catch any
        final_osd_count = extras.get("warnings", {}).get("osd_fail_count", 0)
        if final_osd_count == 0 and osd_fail_count > 0:
            extras.setdefault("warnings", {})["osd_fail_count"] = osd_fail_count
        
        # Normalize OCR engine name
        if 'result' in locals() and hasattr(converter, 'format_options') and fmt == "pdf":
            pdf_format = converter.format_options.get(InputFormat.PDF)
            if pdf_format and hasattr(pdf_format, 'pipeline_options'):
                ocr_opts = pdf_format.pipeline_options.ocr_options
                extras["ocr_engine"] = _normalize_ocr_engine_name(ocr_opts)

        # Check result object
        if 'result' in locals():
            print(f"Phase 2: Checking result object...")
            print(f"  - result type: {type(result)}")
            print(f"  - has document: {hasattr(result, 'document')}")
            if hasattr(result, 'document'):
                print(f"  - document type: {type(result.document)}")
                print(f"  - document has pages: {hasattr(result.document, 'pages')}")
                if hasattr(result.document, 'pages'):
                    print(f"  - total pages: {len(result.document.pages)}")
            print(f"Phase 2 complete in {time.time() - start_time:.2f}s")

        process_seconds = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"CONVERSION COMPLETE: Total time {process_seconds:.2f}s")
        print(f"{'='*60}\n")

        # Check if processing was aborted
        if extras.get("abort"):
            st.warning(f"⚠️ Processing aborted: {extras['abort']['reason']}")
        else:
            st.success("Document processed successfully!")

        # 3. Display the results
        st.subheader("Extracted Content (Markdown)")

        # Export the document's content to Markdown (if not aborted)
        if 'result' in locals() and result is not None and hasattr(result, 'document'):
            print("Phase 3: Exporting to markdown...")
            export_start = time.time()
            # Check time limit before export
            if time.time() - start_time > MAX_SECONDS:
                extras["abort"] = {"reason": "MAX_SECONDS", "limit": MAX_SECONDS, "actual": time.time() - start_time}
                markdown_output = ""
            else:
                markdown_output = result.document.export_to_markdown()
                print(f"Phase 3 complete: export_to_markdown() in {time.time() - export_start:.2f}s")
                print(f"  - Markdown length: {len(markdown_output)} characters")
        else:
            markdown_output = ""
        
        # Extract comprehensive metrics
        metrics = _extract_document_metrics(result if 'result' in locals() and result is not None else None, markdown_output, fmt, process_seconds, extras)
        
        # Add correlation IDs to metrics for MongoDB storage
        metrics["run_id"] = run_id
        metrics["content_hash"] = content_hash
        
        # Markdown runaway guard: check before storing
        if metrics["metrics"]["markdown_length"] > 2_000_000:
            metrics["status"]["quality_bucket"] = "suspect"
            logging.warning(_scrub_secrets(f"WARNING - OVERSIZE: markdown_length={metrics['metrics']['markdown_length']} exceeds 2M chars run_id={run_id} hash={content_hash[:8]}"))
        
        # Log structured QA line with correlation IDs
        page_count = metrics["metrics"].get("page_count", "?")
        md_len = metrics["metrics"]["markdown_length"]
        osd_fails = metrics["warnings"]["osd_fail_count"]
        wmf_skipped = metrics.get("rasterized_graphics_skipped", 0)
        tlayer = metrics.get("text_layer_detected", False)
        bucket = metrics["status"]["quality_bucket"]
        abort_reason = metrics["status"].get("abort", {}).get("reason", "") if metrics["status"].get("abort") else ""
        osd_collapsed = 1 if extras.get("osd_suppressed_after_first") else 0
        
        logging.info(_scrub_secrets(
            f"QA: format={fmt.upper()} pages={page_count} md={md_len} "
            f"osd_fails={osd_fails} wmf_skipped={wmf_skipped} tlayer={tlayer} "
            f"osd_collapsed={osd_collapsed} bucket={bucket} sec={process_seconds:.2f} run_id={run_id} hash={content_hash[:8]}"
            + (f" abort={abort_reason}" if abort_reason else "")
        ))
        
        # Log normalized warning codes (if flag enabled)
        if QA_FLAG_LOG_NORMALIZED_CODES:
            if metrics["warnings"].get("wmf_missing_loader"):
                logging.warning(_scrub_secrets(f"PPTX: WMF graphic skipped (WMF_LOADER_MISSING) run_id={run_id} hash={content_hash[:8]}"))
            if osd_fails > 0:
                if extras.get("osd_suppressed_after_first"):
                    logging.warning(_scrub_secrets(f"PDF: OSD suppressed after first failure on page 0 (OSD_FAIL_COLLAPSED) run_id={run_id} hash={content_hash[:8]}"))
                else:
                    logging.warning(_scrub_secrets(f"PDF: OSD failures detected: {osd_fails} (OSD_FAIL) run_id={run_id} hash={content_hash[:8]}"))
            if md_len < 500:
                logging.warning(_scrub_secrets(f"Document: Short markdown output: {md_len} chars (SHORT_MD) run_id={run_id} hash={content_hash[:8]}"))
            if metrics["warnings"].get("format_conflict"):
                logging.warning(_scrub_secrets(f"Document: Format conflict detected (FORMAT_CONFLICT) run_id={run_id} hash={content_hash[:8]}"))

        # Display the Markdown in the app.
        # We use a text_area for long outputs, but st.markdown() also works.
        # Unique key prevents conflicts when multiple files are processed
        st.text_area("Markdown Output", markdown_output, height=600, key=f"output_{uploaded_file.name}")

        # 4. Download functionality - enabled by default
        st.divider()
        st.subheader("Download Processed Document")
        
        if ENABLE_DOWNLOADS:
            st.write("Download the processed document in various formats:")

            # Get text normalization options from session state
            apply_normalization = st.session_state.get("apply_text_normalization", False)
            fix_spacing = st.session_state.get("fix_spacing", True)
            fix_ligatures = st.session_state.get("fix_ligatures", True)
            filter_ocr = st.session_state.get("filter_ocr_artifacts", False)

            col1, col2, col3, col4, col5 = st.columns(5)

            try:
                base_filename = os.path.splitext(uploaded_file.name)[0] if uploaded_file.name else "document"

                # Markdown download
                with col1:
                    md_data, md_filename = _prepare_download_data(
                        result, "markdown", base_filename,
                        normalize_text=apply_normalization,
                        fix_spacing=fix_spacing,
                        fix_ligatures=fix_ligatures,
                        filter_ocr_artifacts=filter_ocr
                    )
                    st.download_button(
                        label="Download Markdown",
                        data=md_data,
                        file_name=md_filename,
                        mime="text/markdown",
                        key=f"download_md_{uploaded_file.name}",
                        use_container_width=True
                    )

                # HTML download (includes MathML for formulas)
                with col2:
                    html_data, html_filename = _prepare_download_data(
                        result, "html", base_filename,
                        normalize_text=apply_normalization,
                        fix_spacing=fix_spacing,
                        fix_ligatures=fix_ligatures,
                        filter_ocr_artifacts=filter_ocr
                    )
                    st.download_button(
                        label="Download HTML",
                        data=html_data,
                        file_name=html_filename,
                        mime="text/html",
                        key=f"download_html_{uploaded_file.name}",
                        use_container_width=True
                    )

                # JSON download
                with col3:
                    json_data, json_filename = _prepare_download_data(result, "json", base_filename)
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=json_filename,
                        mime="application/json",
                        key=f"download_json_{uploaded_file.name}",
                        use_container_width=True
                    )

                # Plain text download
                with col4:
                    txt_data, txt_filename = _prepare_download_data(
                        result, "txt", base_filename,
                        normalize_text=apply_normalization,
                        fix_spacing=fix_spacing,
                        fix_ligatures=fix_ligatures,
                        filter_ocr_artifacts=filter_ocr
                    )
                    st.download_button(
                        label="Download Text",
                        data=txt_data,
                        file_name=txt_filename,
                        mime="text/plain",
                        key=f"download_txt_{uploaded_file.name}",
                        use_container_width=True
                    )

                # Doctags download
                with col5:
                    doctags_data, doctags_filename = _prepare_download_data(
                        result, "doctags", base_filename,
                        normalize_text=apply_normalization,
                        fix_spacing=fix_spacing,
                        fix_ligatures=fix_ligatures,
                        filter_ocr_artifacts=filter_ocr
                    )
                    st.download_button(
                        label="Download Doctags",
                        data=doctags_data,
                        file_name=doctags_filename,
                        mime="text/plain",
                        key=f"download_doctags_{uploaded_file.name}",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"ERROR: Download feature unavailable: {e}")
                st.exception(e)
                # App continues normally - download failure doesn't break the app
        else:
            st.warning("Warning: Downloads are disabled in this deployment.")
            st.info("Tip: To enable downloads, set environment variable `ENABLE_DOWNLOADS=true`")

        # 5. MongoDB Storage (if enabled)
        st.divider()
        st.subheader("Database Storage")
        
        if ENABLE_MONGODB:
            if st.session_state.get("mongodb_enabled", False):
                # Check if MongoDB is properly configured
                use_remote = st.session_state.get("use_remote_embeddings", False)
                has_api_key = bool(st.session_state.get("voyageai_api_key", ""))
                has_connection_string = bool(st.session_state.get("mongodb_connection_string", ""))
                is_configured = has_connection_string and (
                    (use_remote and has_api_key) or (not use_remote)
                )

                if is_configured:
                    col_save, col_info = st.columns([1, 2])
                    with col_save:
                        if st.button("Save to MongoDB", key=f"save_mongodb_{uploaded_file.name}", type="primary", use_container_width=True):
                            try:
                                with st.spinner("Saving document to MongoDB with embeddings..."):
                                    import time
                                    storage_start_time = time.time()
                                    
                                    embedding_config = {
                                        "use_remote": st.session_state.get("use_remote_embeddings", False),
                                        "model_name": st.session_state.get("embedding_model_name", DEFAULT_EMBEDDING_MODEL)
                                    }
                                    if st.session_state.get("use_remote_embeddings", False):
                                        embedding_config["api_key"] = st.session_state.get("voyageai_api_key", "")

                                    success = _store_in_mongodb(
                                        document=result.document,
                                        original_filename=uploaded_file.name,
                                        file_size=len(uploaded_file.getvalue()),
                                        mongodb_connection_string=st.session_state.get("mongodb_connection_string", ""),
                                        database_name=st.session_state.get("mongodb_database", "docling_db"),
                                        collection_name=st.session_state.get("mongodb_collection", "documents"),
                                        embedding_config=embedding_config,
                                        metrics=metrics
                                    )
                                    
                                    storage_time = time.time() - storage_start_time
                                    logging.info(f"MongoDB storage completed in {storage_time:.2f}s")
                                    
                                    if success:
                                        st.success("Document saved to MongoDB successfully!")
                                        st.info("Tip: The vector search index is being created automatically. It may take a few minutes to be ready for queries.")
                                    else:
                                        st.error("ERROR: Failed to save document to MongoDB. Check error messages above.")
                            except ImportError as e:
                                st.error(f"ERROR: Missing dependency: {e}")
                                if st.session_state.use_remote_embeddings:
                                    st.info("Tip: Install MongoDB dependencies with: pip install pymongo[srv] voyageai")
                                else:
                                    st.info("Tip: Install MongoDB dependencies with: pip install pymongo[srv] sentence-transformers")
                            except ValueError as e:
                                st.error(f"ERROR: Configuration error: {e}")
                            except Exception as e:
                                error_msg = str(e)
                                st.error(f"ERROR: Error saving to MongoDB: {error_msg}")

                                # Provide helpful troubleshooting for common errors
                                if "SSL" in error_msg or "TLS" in error_msg or "TLSV1_ALERT_INTERNAL_ERROR" in error_msg:
                                    st.warning("**MongoDB Atlas SSL/TLS Connection Issue**")
                                    st.info("""
**CRITICAL: The 'TLSV1_ALERT_INTERNAL_ERROR' is often MISLEADING!**

This error usually means your IP address is NOT whitelisted in MongoDB Atlas, despite appearing as an SSL error.

**PRIMARY SOLUTION (Most Common Cause):**

1. **Check MongoDB Atlas Network Access (IP Whitelisting)**
   - Go to MongoDB Atlas Dashboard
   - Navigate to: Security → Network Access
   - Click "Add IP Address"
   - Add: `0.0.0.0/0` (allow all IPs)
   - **CRITICAL for Hugging Face Spaces**: You MUST use `0.0.0.0/0` because Spaces uses dynamic IPs
   - Wait 1-2 minutes for changes to propagate
   - Try connecting again

**If IP is already whitelisted (0.0.0.0/0), check these:**

2. **Verify Connection String Format**
   - Use `mongodb+srv://` format (NOT `mongodb://`)
   - Example: `mongodb+srv://<username>:<password>@cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority`

3. **Verify Database User Permissions**
   - In Atlas: Security → Database Access
   - Ensure user has "Atlas admin" or "Read and write to any database" role
   - Check username/password are correct in connection string (no special characters unescaped)

4. **Test Connection String**
   - Copy connection string from Atlas: Database → Connect → Drivers → Python
   - Replace `<password>` with actual password (remove `<` `>` brackets)
   - If password has special characters, URL-encode them

**Technical Note:**
Hugging Face Spaces IP addresses change with each deployment. MongoDB Atlas blocks non-whitelisted IPs BEFORE SSL negotiation completes, causing this misleading SSL error.
                                    """)

                                st.exception(e)
                                # App continues normally - storage failure doesn't break the app
                    with col_info:
                        st.info("Tip: Stored documents include full markdown, JSON, and vector embeddings for RAG search.")
                else:
                    st.warning("MongoDB storage is enabled but not fully configured.")
                    if not has_connection_string:
                        st.error("Missing: MongoDB connection string")
                    if use_remote and not has_api_key:
                        st.error("Missing: VoyageAI API key (required for remote embeddings)")
                    st.info("Configure the missing values in the sidebar under 'Configuration'.")
            else:
                st.info("Info: MongoDB storage is available but not enabled.")
                st.info("Tip: Enable it in the sidebar under 'Configuration' → 'MongoDB Storage'")
                if not PYMONGO_AVAILABLE:
                    st.warning("Warning: MongoDB dependencies not installed. Install with: `pip install pymongo[srv] sentence-transformers`")
        else:
            st.info("Info: MongoDB storage feature is not enabled in this deployment.")
            st.info("Tip: To enable: Set environment variable `ENABLE_MONGODB=true` and install dependencies: `pymongo[srv] sentence-transformers`")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
        # Show full traceback in Streamlit for debugging
        st.exception(e)

    finally:
        # Clean up the temporary file
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                os.remove(tmp_file_path)
            except Exception:
                pass  # Ignore cleanup errors
