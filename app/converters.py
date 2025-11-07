"""Document converter management and caching.

Module 1 - Osprey Backend: Document processing application.
"""

import streamlit as st
from app.config import get_settings

# Import will be done here to avoid circular imports
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions, TesseractCliOcrOptions
    from docling.datamodel.settings import settings as docling_settings
    from docling.document_converter import DocumentConverter, PdfFormatOption
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    DocumentConverter = None


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
    if not DOCLING_AVAILABLE:
        raise ImportError("Docling not available. Install with: pip install docling")
    
    app_config = get_settings()
    
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True

    # Choose Tesseract OCR engine based on environment
    # If TESSDATA_PREFIX is set and available, use the library version (faster)
    # Otherwise, use the CLI version (auto-detects language data, graceful degradation)
    import os
    ocr_available = bool(app_config.tessdata_prefix and os.path.exists(app_config.tessdata_prefix))
    if ocr_available:
        pipeline_options.ocr_options = TesseractOcrOptions()
    else:
        pipeline_options.ocr_options = TesseractCliOcrOptions()

    # Force CPU device to avoid GPU/CUDA/MPS detection issues
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU

    # Enable enrichment features
    pipeline_options.do_formula_enrichment = enable_formula_enrichment
    pipeline_options.do_table_structure = enable_table_structure
    pipeline_options.do_code_enrichment = enable_code_enrichment
    pipeline_options.do_picture_classification = enable_picture_classification

    # Image generation is always enabled with optimized settings
    # This reduces memory pressure while maintaining image functionality
    pipeline_options.do_image_generation = True
    pipeline_options.image_generation_scale = app_config.images_scale

    # Configure global Docling settings
    docling_settings.perf.page_batch_size = app_config.page_batch_size
    docling_settings.artifacts_path = None

    converter = DocumentConverter(
        format_options={
            PdfFormatOption.STANDARD: pipeline_options
        }
    )

    return converter

