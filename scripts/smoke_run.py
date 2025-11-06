#!/usr/bin/env python3
"""
Smoke test script to run conversions on sample files and verify MongoDB document structure.

This script:
1. Runs conversion on each file in smoke/ folder
2. Prints the structured QA log line
3. Verifies Mongo doc fields exist (input, metrics, warnings, status)
4. Exits non-zero on missing fields
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to import app functions
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app import (
        sniff_file_format,
        _extract_document_metrics,
        get_converter,
    )
    from docling.datamodel.base_models import InputFormat
except ImportError as e:
    print(f"ERROR: Failed to import app modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)


def verify_metrics_structure(metrics: dict, filename: str) -> bool:
    """
    Verify that metrics dictionary has all required fields.
    
    Args:
        metrics: Metrics dictionary from _extract_document_metrics
        filename: Original filename for error reporting
        
    Returns:
        True if all fields present, False otherwise
    """
    required_fields = {
        "input": {"format"},
        "metrics": {"markdown_length", "process_seconds"},
        "warnings": {"osd_fail_count", "wmf_missing_loader", "format_conflict"},
        "status": {"quality_bucket"},
    }
    
    missing = []
    
    # Check top-level fields
    for field, subfields in required_fields.items():
        if field not in metrics:
            missing.append(f"{field} (missing)")
            continue
        
        if isinstance(subfields, set):
            for subfield in subfields:
                if subfield not in metrics[field]:
                    missing.append(f"{field}.{subfield}")
    
    # Check optional but important fields
    optional_fields = ["text_layer_detected", "rasterized_graphics_skipped", "schema_version"]
    for field in optional_fields:
        if field not in metrics:
            # These are optional but should be present in new docs
            pass
    
    if missing:
        print(f"  ✗ Missing fields in {filename}: {', '.join(missing)}")
        return False
    
    print(f"  ✓ All required fields present")
    return True


def process_smoke_file(filepath: Path, converter) -> bool:
    """
    Process a single smoke test file.
    
    Args:
        filepath: Path to file to process
        converter: DocumentConverter instance
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Processing: {filepath.name}")
    print(f"{'='*60}")
    
    try:
        # Read file
        with open(filepath, 'rb') as f:
            file_bytes = f.read()
            first_bytes = file_bytes[:8]
        
        # Sniff format
        fmt, confidence, conflict, ext_guess = sniff_file_format(filepath.name, first_bytes)
        print(f"  Format: {fmt} (confidence: {confidence:.2f}, conflict: {conflict})")
        
        # Initialize extras
        extras = {
            "warnings": {},
            "rasterized_graphics_skipped": 0,
            "text_layer_detected": False,
            "ocr_engine": "unknown"
        }
        
        if conflict:
            extras["warnings"]["format_conflict"] = True
        
        # Run conversion
        import time
        start_time = time.time()
        
        try:
            result = converter.convert(str(filepath))
            process_seconds = time.time() - start_time
            
            # Export markdown
            markdown_output = result.document.export_to_markdown()
            
            # Extract metrics
            metrics = _extract_document_metrics(
                result,
                markdown_output,
                fmt,
                process_seconds,
                extras
            )
            
            # Print QA log line
            page_count = metrics["metrics"].get("page_count", "?")
            md_len = metrics["metrics"]["markdown_length"]
            osd_fails = metrics["warnings"]["osd_fail_count"]
            wmf_skipped = metrics.get("rasterized_graphics_skipped", 0)
            tlayer = metrics.get("text_layer_detected", False)
            bucket = metrics["status"]["quality_bucket"]
            
            print(f"  QA: format={fmt.upper()} pages={page_count} md={md_len} "
                  f"osd_fails={osd_fails} wmf_skipped={wmf_skipped} tlayer={tlayer} "
                  f"bucket={bucket} sec={process_seconds:.2f}")
            
            # Verify structure
            return verify_metrics_structure(metrics, filepath.name)
            
        except Exception as e:
            print(f"  ✗ Conversion failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"  ✗ File read failed: {e}")
        return False


def main():
    """Main smoke test function."""
    smoke_dir = Path(__file__).parent.parent / "smoke"
    
    if not smoke_dir.exists():
        print(f"ERROR: smoke/ directory not found at {smoke_dir}")
        print("Create smoke/ directory and add sample files:")
        print("  - sample.pdf (text layer)")
        print("  - scan_cover.pdf (image cover page only)")
        print("  - sample.pptx (1-2 slides with WMF image placeholder)")
        print("  - sample.xlsx (one sheet table)")
        sys.exit(1)
    
    # Find sample files
    sample_files = []
    for pattern in ["*.pdf", "*.pptx", "*.xlsx"]:
        sample_files.extend(smoke_dir.glob(pattern))
    
    if not sample_files:
        print(f"WARNING: No sample files found in {smoke_dir}")
        print("Add sample files to run smoke tests")
        sys.exit(0)
    
    print(f"Found {len(sample_files)} sample file(s)")
    
    # Initialize converter
    print("\nInitializing converter...")
    try:
        converter = get_converter(
            enable_formula_enrichment=False,
            enable_table_structure=True,
            enable_code_enrichment=False,
            enable_picture_classification=False
        )
        print("✓ Converter initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize converter: {e}")
        sys.exit(1)
    
    # Process each file
    results = []
    for filepath in sorted(sample_files):
        success = process_smoke_file(filepath, converter)
        results.append((filepath.name, success))
    
    # Summary
    print(f"\n{'='*60}")
    print("SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for filename, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {filename}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    if passed < total:
        sys.exit(1)
    else:
        print("\n✓ All smoke tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

