# Smoke Test Files

This directory contains sample files for smoke testing the document conversion pipeline.

## Required Files

Add the following files to this directory:

1. **sample.pdf** - A PDF with text layer (for testing text extraction)
2. **scan_cover.pdf** - A scanned PDF with image cover page only (for testing OCR)
3. **sample.pptx** - A PowerPoint file with 1-2 slides containing a WMF image placeholder (for testing WMF handling)
4. **sample.xlsx** - An Excel file with one sheet containing a table (for testing table extraction)

## Usage

Run the smoke test script:

```bash
python scripts/smoke_run.py
```

The script will:
- Process each file
- Print structured QA log lines
- Verify MongoDB document structure
- Exit with non-zero code if any test fails

## Note

These files are not included in the repository. You need to add them manually for smoke testing.

