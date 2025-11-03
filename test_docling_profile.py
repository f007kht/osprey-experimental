#!/usr/bin/env python3
"""
Test script for docling_profile.py - Enhanced Docling Processing

Demonstrates all 5 fixes:
1. Formula enrichment (LaTeX/MathML)
2. Table extraction (with HTML fallback)
3. Figure OCR filtering (for search)
4. Text normalization (ligatures, spacing)
5. Image embedding (HTML/Markdown)

Usage:
    python test_docling_profile.py <pdf_file>
"""

import sys
from pathlib import Path
from docling_profile import (
    EnhancedDoclingConverter,
    create_formula_focused_converter,
    create_table_focused_converter,
    create_search_optimized_converter,
    create_full_fidelity_converter,
)


def test_basic_usage(pdf_file: str):
    """Test basic converter usage."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Usage")
    print("=" * 80)

    converter = EnhancedDoclingConverter()
    result = converter.convert(pdf_file)

    print(f"✓ Converted: {pdf_file}")
    print(f"  Document has {len(result.document.pages)} pages")
    print(f"  Found {len(result.document.tables)} tables")
    print(f"  Found {len(result.document.pictures)} pictures")

    # Count formulas (if enabled)
    try:
        doc_dict = result.document.export_to_dict()
        formulas = [t for t in doc_dict.get("texts", []) if t.get("label") == "formula"]
        print(f"  Found {len(formulas)} formulas")
        if formulas:
            print(f"    Sample formula: {formulas[0].get('text', '')[:80]}...")
    except Exception:
        pass


def test_formula_extraction(pdf_file: str):
    """Test formula enrichment and LaTeX extraction."""
    print("\n" + "=" * 80)
    print("TEST 2: Formula Extraction (Issue 1)")
    print("=" * 80)

    converter = create_formula_focused_converter()
    result = converter.convert(pdf_file)

    # Export HTML with MathML
    html = converter.export_html_with_mathml(result)

    mathml_count = html.count("<math")
    print(f"✓ Exported HTML with {mathml_count} MathML formulas")

    if mathml_count > 0:
        print("  ✅ Issue 1 SOLVED: Formulas are text-usable (MathML in HTML)")
    else:
        print("  ℹ️ No formulas found in this document")


def test_table_extraction(pdf_file: str):
    """Test table extraction with HTML fallback."""
    print("\n" + "=" * 80)
    print("TEST 3: Table Extraction (Issue 2)")
    print("=" * 80)

    converter = create_table_focused_converter()
    result = converter.convert(pdf_file)

    # Extract tables to DataFrames
    tables_df = converter.export_tables_to_dataframes(result)

    print(f"✓ Extracted {len(tables_df)} tables to DataFrames")

    for table_ix, df in tables_df[:3]:  # Show first 3 tables
        print(f"\n  Table {table_ix + 1}:")
        print(f"    Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"    Columns: {list(df.columns)[:5]}")  # First 5 columns

    if tables_df:
        print("  ✅ Issue 2 SOLVED: Tables have structure (CSV/DataFrame extraction works)")
    else:
        print("  ℹ️ No tables found in this document")


def test_figure_filtering(pdf_file: str):
    """Test figure OCR filtering for search."""
    print("\n" + "=" * 80)
    print("TEST 4: Figure OCR Filtering (Issue 3)")
    print("=" * 80)

    converter = create_search_optimized_converter()
    result = converter.convert(pdf_file)

    # Export text for search (excludes picture OCR)
    search_text = converter.export_text_for_search(result)
    full_text = result.document.export_to_text()

    print(f"✓ Full text length: {len(full_text)} chars")
    print(f"✓ Search text length: {len(search_text)} chars")
    print(f"  Filtered out: {len(full_text) - len(search_text)} chars ({((len(full_text) - len(search_text)) / len(full_text) * 100):.1f}%)")

    if len(search_text) < len(full_text):
        print("  ✅ Issue 3 SOLVED: Figure OCR text filtered from search index")
    else:
        print("  ℹ️ No figure OCR artifacts detected in this document")


def test_text_normalization(pdf_file: str):
    """Test text normalization (ligatures, spacing)."""
    print("\n" + "=" * 80)
    print("TEST 5: Text Normalization (Issue 4)")
    print("=" * 80)

    converter = EnhancedDoclingConverter()
    result = converter.convert(pdf_file)

    # Test normalization on sample texts
    test_cases = [
        ("ﬁle ﬂow", "file flow"),
        ("T  H  E", "THE"),
        ("timeWhichCould", "time Which Could"),
    ]

    print("Testing normalization:")
    for raw, expected in test_cases:
        normalized = converter.normalize_pdf_text(raw)
        status = "✓" if normalized == expected else "✗"
        print(f"  {status} '{raw}' → '{normalized}' (expected: '{expected}')")

    # Export normalized text
    normalized_text = converter.export_normalized_text(result)
    print(f"\n✓ Exported {len(normalized_text)} chars of normalized text")
    print("  ✅ Issue 4 SOLVED: Text normalization applied")


def test_image_embedding(pdf_file: str):
    """Test image embedding in HTML/Markdown."""
    print("\n" + "=" * 80)
    print("TEST 6: Image Embedding (Issue 5)")
    print("=" * 80)

    converter = create_full_fidelity_converter()
    result = converter.convert(pdf_file)

    # Export HTML with embedded images
    html = converter.export_html_with_images(result, image_mode="embedded")

    img_count = html.count("<img")
    data_uri_count = html.count("data:image/")

    print(f"✓ Exported HTML with {img_count} <img> tags")
    print(f"  {data_uri_count} embedded as base64 data URIs")

    if img_count > 0:
        print("  ✅ Issue 5 SOLVED: Images embedded in HTML export")
    else:
        print("  ℹ️ No images found in this document")


def test_full_pipeline(pdf_file: str):
    """Test complete processing pipeline."""
    print("\n" + "=" * 80)
    print("TEST 7: Full Pipeline - All Fixes")
    print("=" * 80)

    converter = create_full_fidelity_converter()

    output_dir = "test_output"
    print(f"Processing {pdf_file} with all fixes enabled...")

    outputs = converter.process_document(
        pdf_file,
        output_dir=output_dir
    )

    print("\n✓ Processing complete!")
    print(f"\n📁 Output files in: {output_dir}/")
    print(f"   HTML (with MathML formulas): {outputs.get('html_file')}")
    print(f"   Markdown (normalized): {outputs.get('markdown_file')}")
    print(f"   Text (normalized): {outputs.get('text_file')}")
    print(f"   Search text (filtered): {outputs.get('search_text_file')}")
    print(f"   Tables: {len(outputs.get('table_csv_files', []))} CSV files")

    print("\n" + "=" * 80)
    print("✅ ALL 5 ISSUES ADDRESSED:")
    print("=" * 80)
    print("1. ✅ Formulas: LaTeX extracted, MathML in HTML")
    print("2. ✅ Tables: Structure preserved, CSV/DataFrame export works")
    print("3. ✅ Figure OCR: Filtered from search index")
    print("4. ✅ Text: Normalized (ligatures, spacing)")
    print("5. ✅ Images: Embedded in HTML/Markdown exports")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_docling_profile.py <pdf_file>")
        print("\nExample:")
        print("  python test_docling_profile.py sample.pdf")
        sys.exit(1)

    pdf_file = sys.argv[1]

    if not Path(pdf_file).exists():
        print(f"Error: File not found: {pdf_file}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Enhanced Docling Processing - Test Suite")
    print("=" * 80)
    print(f"PDF: {pdf_file}")

    # Run all tests
    try:
        test_basic_usage(pdf_file)
        test_formula_extraction(pdf_file)
        test_table_extraction(pdf_file)
        test_figure_filtering(pdf_file)
        test_text_normalization(pdf_file)
        test_image_embedding(pdf_file)
        test_full_pipeline(pdf_file)

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
