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
from typing import Optional, List, Dict, Any
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


def sniff_file_format(file_path: str) -> Optional[str]:
    """
    Detect file format by checking magic number (first 4 bytes).
    
    This helps identify Office files (XLSX/DOCX/PPTX) that have PK header
    before they're incorrectly detected as PDF, reducing false warnings.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Detected format string ('pdf', 'xlsx', 'docx', 'pptx', 'zip-office', or None)
    """
    try:
        from pathlib import Path
        sig = Path(file_path).read_bytes()[:4]
        
        if sig.startswith(b"%PDF"):
            return "pdf"
        if sig.startswith(b"PK\x03\x04"):
            # ZIP container - could be Office file
            low = file_path.lower()
            if low.endswith(".xlsx"):
                return "xlsx"
            if low.endswith(".docx"):
                return "docx"
            if low.endswith(".pptx"):
                return "pptx"
            return "zip-office"
        return None
    except Exception:
        # If we can't read the file, return None (let docling handle it)
        return None


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
            
            # Insert primary document with retry
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
            # Document fits in single BSON doc - insert normally with retry
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

    # Sniff file format by magic number to help with logging
    detected_format = sniff_file_format(tmp_file_path)
    if detected_format:
        logging.debug(f"File signature detected: {detected_format} for {uploaded_file.name}")

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

        # Run the conversion process
        with st.spinner(f"Converting `{uploaded_file.name}`..."):
            import time
            start_time = time.time()
            print(f"\n{'='*60}")
            print(f"CONVERSION START: {uploaded_file.name}")
            print(f"{'='*60}")

            # Option B: Detect document size and use chunked processing for large documents
            # This prevents the 127-page stopping issue by processing in manageable chunks
            try:
                from pypdf import PdfReader
                pdf_reader = PdfReader(tmp_file_path)
                total_pages = len(pdf_reader.pages)
                print(f"Document has {total_pages} pages")

                # For documents > 120 pages, process in chunks to avoid memory/timeout issues
                MAX_PAGES_PER_CHUNK = 120
                if total_pages > MAX_PAGES_PER_CHUNK:
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

            except Exception as e:
                # Fallback to normal processing if page detection fails
                print(f"Could not detect page count: {e}. Using normal processing.")
                print("Phase 1: Starting converter.convert()...")
                result = converter.convert(tmp_file_path)
                print(f"Phase 1 complete: converter.convert() returned in {time.time() - start_time:.2f}s")

            # Check result object
            print(f"Phase 2: Checking result object...")
            print(f"  - result type: {type(result)}")
            print(f"  - has document: {hasattr(result, 'document')}")
            if hasattr(result, 'document'):
                print(f"  - document type: {type(result.document)}")
                print(f"  - document has pages: {hasattr(result.document, 'pages')}")
                if hasattr(result.document, 'pages'):
                    print(f"  - total pages: {len(result.document.pages)}")
            print(f"Phase 2 complete in {time.time() - start_time:.2f}s")

        print(f"\n{'='*60}")
        print(f"CONVERSION COMPLETE: Total time {time.time() - start_time:.2f}s")
        print(f"{'='*60}\n")

        st.success("Document processed successfully!")

        # 3. Display the results
        st.subheader("Extracted Content (Markdown)")

        # Export the document's content to Markdown
        print("Phase 3: Exporting to markdown...")
        export_start = time.time()
        markdown_output = result.document.export_to_markdown()
        print(f"Phase 3 complete: export_to_markdown() in {time.time() - export_start:.2f}s")
        print(f"  - Markdown length: {len(markdown_output)} characters")

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
                                        embedding_config=embedding_config
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
