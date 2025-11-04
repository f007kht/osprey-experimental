#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Monitoring Utilities for Docling Processing

This module provides utilities to monitor memory usage during document processing
with image generation enabled. Use these tools to track memory consumption and
identify potential memory bottlenecks.

Usage:
    python memory_monitor.py <pdf_file_path>

Example:
    python memory_monitor.py test_document.pdf
"""

import os
import sys
import time
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with: pip install psutil")

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.settings import settings
    DOCLING_AVAILABLE = True
except ImportError as e:
    print(f"Error: Failed to import docling modules: {e}")
    sys.exit(1)


def get_memory_usage():
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB


def print_memory(label: str, previous_mb: float = None):
    """Print memory usage with optional delta."""
    if not PSUTIL_AVAILABLE:
        print(f"{label}: (psutil not available)")
        return None
    
    current_mb = get_memory_usage()
    if previous_mb is not None:
        delta = current_mb - previous_mb
        print(f"{label}: {current_mb:.2f} MB (Δ {delta:+.2f} MB)")
    else:
        print(f"{label}: {current_mb:.2f} MB")
    
    return current_mb


def monitor_conversion(file_path: str):
    """Monitor memory during document conversion."""
    print(f"\n{'='*60}")
    print(f"Memory Monitor for Docling Document Processing")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    print(f"File size: {Path(file_path).stat().st_size / 1024:.2f} KB")
    print(f"{'='*60}\n")
    
    # Configure pipeline with memory optimizations
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = TesseractOcrOptions()
    
    # Enable image generation (as per requirements)
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 1.0  # Reduced to 1.0 for large document support
    
    # Memory optimizations
    pipeline_options.queue_max_size = 10
    pipeline_options.ocr_batch_size = 1
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    
    # Global settings
    settings.perf.page_batch_size = 1
    
    # Build converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # Monitor memory at each stage
    mem_before = print_memory("Memory before conversion")
    time.sleep(0.1)  # Small delay for measurement
    
    start_time = time.time()
    result = converter.convert(file_path)
    conversion_time = time.time() - start_time
    
    mem_after = print_memory("Memory after conversion", mem_before)
    
    # Get document info
    doc = result.document
    page_count = len(result.document.pages)
    print(f"\nConversion completed:")
    print(f"  - Pages processed: {page_count}")
    print(f"  - Conversion time: {conversion_time:.2f} seconds")
    print(f"  - Memory increase: {mem_after - mem_before:.2f} MB")
    print(f"  - Memory per page: {(mem_after - mem_before) / page_count:.2f} MB/page")
    
    # Test accessing images
    if page_count > 0:
        print(f"\n{'='*60}")
        print("Testing image access (first 5 pages):")
        print(f"{'='*60}")
        
        mem_before_images = get_memory_usage()
        for i, page in enumerate(doc.pages[:5]):
            try:
                img = page.get_image()
                mem_after_page = get_memory_usage()
                delta = mem_after_page - mem_before_images
                print(f"  Page {i+1}: Image size {img.size if img else 'N/A'}, "
                      f"Memory: {mem_after_page:.2f} MB (Δ {delta:+.2f} MB)")
                mem_before_images = mem_after_page
            except Exception as e:
                print(f"  Page {i+1}: Error accessing image - {e}")
    
    # Final memory
    mem_final = print_memory("\nFinal memory usage", mem_before)
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  - Peak memory: {mem_final:.2f} MB")
    print(f"  - Total increase: {mem_final - mem_before:.2f} MB")
    print(f"{'='*60}\n")
    
    return result


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python memory_monitor.py <pdf_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not Path(file_path).exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    try:
        monitor_conversion(file_path)
    except Exception as e:
        print(f"\nError during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

