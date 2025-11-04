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
from typing import Optional

try:
    import psutil
except ImportError:
    print("ERROR: psutil is required for memory monitoring.")
    print("Install with: pip install psutil")
    sys.exit(1)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def print_memory(label: str, start_mb: Optional[float] = None):
    """
    Print current memory usage with optional delta from start.

    Args:
        label: Description of the current stage
        start_mb: Optional starting memory to calculate delta
    """
    current_mb = get_memory_mb()
    if start_mb is not None:
        delta_mb = current_mb - start_mb
        print(f"{label}: {current_mb:.2f} MB (Δ {delta_mb:+.2f} MB)")
    else:
        print(f"{label}: {current_mb:.2f} MB")


def build_optimized_converter(
    enable_ocr: bool = True,
    enable_formula_enrichment: bool = True,
    enable_table_structure: bool = True,
    generate_page_images: bool = True,
    generate_picture_images: bool = True,
    images_scale: float = 1.5,
    queue_max_size: int = 10,
    batch_size: int = 1
) -> DocumentConverter:
    """
    Build a DocumentConverter with memory-optimized settings.

    Args:
        enable_ocr: Enable OCR for text extraction
        enable_formula_enrichment: Extract LaTeX formulas
        enable_table_structure: Extract table structures
        generate_page_images: Generate full page images
        generate_picture_images: Generate images for detected pictures
        images_scale: Image scale factor (1.5 recommended for memory optimization)
        queue_max_size: Maximum queue size (10 recommended, default is 100)
        batch_size: Batch size for processing stages (1 recommended for memory optimization)

    Returns:
        Configured DocumentConverter instance
    """
    # Configure global settings
    settings.perf.page_batch_size = batch_size

    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = enable_ocr
    pipeline_options.ocr_options = TesseractOcrOptions()
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU

    # Enrichment features
    pipeline_options.do_formula_enrichment = enable_formula_enrichment
    pipeline_options.do_table_structure = enable_table_structure

    # Image generation settings
    pipeline_options.generate_page_images = generate_page_images
    pipeline_options.generate_picture_images = generate_picture_images
    pipeline_options.images_scale = images_scale

    # Memory optimization settings
    pipeline_options.queue_max_size = queue_max_size
    pipeline_options.ocr_batch_size = batch_size
    pipeline_options.layout_batch_size = batch_size
    pipeline_options.table_batch_size = batch_size

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def process_with_monitoring(file_path: str, max_pages: Optional[int] = None):
    """
    Process a document with detailed memory monitoring.

    Args:
        file_path: Path to the PDF file to process
        max_pages: Optional limit on number of pages to process
    """
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return

    file_size_mb = os.path.getsize(file_path) / 1024 / 1024
    print(f"\n{'='*60}")
    print(f"Processing: {file_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"{'='*60}\n")

    # Track starting memory
    start_mb = get_memory_mb()
    print_memory("Initial memory")

    # Build converter
    print("\nBuilding converter with optimized settings...")
    print("  - Image generation: ENABLED")
    print("  - Image scale: 1.5x")
    print("  - Queue max size: 10 (reduced from 100)")
    print("  - Batch sizes: 1 (process one page at a time)")
    print("  - Page batch size: 1")

    converter = build_optimized_converter()
    print_memory("After converter initialization", start_mb)

    # Process document
    print(f"\nProcessing document...")
    conversion_start = time.time()

    try:
        if max_pages:
            result = converter.convert(file_path, page_range=(1, max_pages))
        else:
            result = converter.convert(file_path)

        conversion_time = time.time() - conversion_start
        print_memory("After conversion", start_mb)

        # Document info
        total_pages = len(result.document.pages)
        print(f"\nConversion completed in {conversion_time:.2f} seconds")
        print(f"Total pages: {total_pages}")
        print(f"Time per page: {conversion_time/total_pages:.2f} seconds")

        # Access images to check memory usage
        print("\nAccessing page images...")
        for i, page in enumerate(result.document.pages[:min(5, total_pages)]):
            img = page.get_image()
            if img:
                print_memory(f"  Page {i+1} image accessed ({img.size})", start_mb)

        # Export to markdown
        print("\nExporting to markdown...")
        markdown = result.document.export_to_markdown()
        print_memory("After markdown export", start_mb)
        print(f"Markdown length: {len(markdown)} characters")

        # Final memory check
        print("\nFinal memory usage:")
        print_memory("Final memory", start_mb)

        # Peak memory
        peak_mb = get_memory_mb()
        peak_delta = peak_mb - start_mb
        print(f"\nPeak memory increase: {peak_delta:.2f} MB")

        # Memory per page estimate
        if total_pages > 0:
            mem_per_page = peak_delta / total_pages
            print(f"Estimated memory per page: {mem_per_page:.2f} MB")

        print(f"\n{'='*60}")
        print("Processing completed successfully!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\nERROR during processing: {e}")
        print_memory("Memory at error", start_mb)
        raise


def main():
    """Main entry point for memory monitoring."""
    if len(sys.argv) < 2:
        print("Usage: python memory_monitor.py <pdf_file_path> [max_pages]")
        print("\nExample:")
        print("  python memory_monitor.py document.pdf")
        print("  python memory_monitor.py document.pdf 10  # Process first 10 pages only")
        sys.exit(1)

    file_path = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else None

    process_with_monitoring(file_path, max_pages)


if __name__ == "__main__":
    main()
