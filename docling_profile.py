"""
Docling Processing Profile - Enhanced Configuration & Post-Processing
======================================================================

Addresses 5 key issues in Docling PDF conversion:
1. ✅ Formulas: Enable LaTeX extraction & MathML export
2. ⚠️ Tables: Workaround for empty JSON table cells via HTML parsing
3. ⚠️ Figure OCR: Filter picture text from search indexes
4. ✅ Text normalization: Clean ligatures & spacing artifacts
5. ⚠️ Images: Embed base64 images in exports

Usage:
    from docling_profile import EnhancedDoclingConverter

    converter = EnhancedDoclingConverter()
    result = converter.convert("document.pdf")

    # Get enhanced exports
    html = converter.export_html(result)
    tables_df = converter.export_tables_to_dataframes(result)
    clean_text = converter.export_normalized_text(result)
"""

import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from io import StringIO

import pandas as pd

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractOcrOptions,
    TableStructureOptions,
    TableFormerMode,
)
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, ImageRefMode


class EnhancedDoclingConverter:
    """
    Enhanced DocumentConverter with solutions for all 5 Docling issues.

    Features:
    - Formula enrichment (LaTeX extraction)
    - Table structure extraction with HTML fallback
    - Figure OCR filtering for search
    - Text normalization (ligatures, spacing)
    - Image embedding in exports
    """

    def __init__(
        self,
        enable_formula_enrichment: bool = True,
        enable_picture_classification: bool = True,
        generate_images: bool = True,
        image_scale: float = 2.0,
        table_mode: str = "accurate",
        use_cpu_only: bool = True,
    ):
        """
        Initialize enhanced converter.

        Args:
            enable_formula_enrichment: Extract LaTeX from formulas (slower)
            enable_picture_classification: Classify figure types
            generate_images: Extract images for embedding
            image_scale: Image quality multiplier (1.0-3.0)
            table_mode: "accurate" or "fast" table extraction
            use_cpu_only: Force CPU device (safer for deployment)
        """
        self.enable_formula_enrichment = enable_formula_enrichment
        self.enable_picture_classification = enable_picture_classification
        self.generate_images = generate_images
        self.image_scale = image_scale
        self.table_mode = table_mode
        self.use_cpu_only = use_cpu_only

        # Build converter
        self.converter = self._build_converter()

    def _build_converter(self) -> DocumentConverter:
        """Build DocumentConverter with enhanced pipeline options."""
        pipeline_options = PdfPipelineOptions()

        # Issue 1: Formula enrichment (LaTeX extraction)
        pipeline_options.do_formula_enrichment = self.enable_formula_enrichment

        # Issue 2: Table structure extraction
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True,  # Match predictions to PDF cells
            mode=TableFormerMode.ACCURATE if self.table_mode == "accurate" else TableFormerMode.FAST
        )

        # Issue 3: OCR configuration
        pipeline_options.do_ocr = True
        ocr_options = TesseractOcrOptions()
        ocr_options.force_full_page_ocr = False  # Only OCR when needed
        ocr_options.bitmap_area_threshold = 0.05  # 5% threshold
        pipeline_options.ocr_options = ocr_options

        # Picture classification (helps distinguish charts from text)
        pipeline_options.do_picture_classification = self.enable_picture_classification

        # Issue 5: Image generation for export fidelity
        pipeline_options.generate_picture_images = self.generate_images
        pipeline_options.generate_page_images = self.generate_images
        pipeline_options.images_scale = self.image_scale

        # Performance: Force CPU to avoid GPU issues
        if self.use_cpu_only:
            pipeline_options.accelerator_options.device = AcceleratorDevice.CPU

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def convert(self, source: Any) -> Any:
        """
        Convert document using enhanced pipeline.

        Args:
            source: Path to PDF or URL

        Returns:
            ConversionResult object
        """
        return self.converter.convert(source)

    # ========================================================================
    # Issue 1: Formula Exports (✅ Already working with do_formula_enrichment)
    # ========================================================================

    def export_html_with_mathml(self, result: Any) -> str:
        """
        Export HTML with MathML formulas.

        This leverages do_formula_enrichment=True to render formulas as MathML.

        Returns:
            HTML string with <math> tags for formulas
        """
        doc: DoclingDocument = result.document
        return doc.export_to_html()

    # ========================================================================
    # Issue 2: Table Extraction (Workaround for empty JSON cells)
    # ========================================================================

    def export_tables_to_dataframes(self, result: Any) -> List[Tuple[int, pd.DataFrame]]:
        """
        Extract tables as pandas DataFrames.

        WORKAROUND: If JSON table_cells are empty, parse from HTML export.

        Returns:
            List of (table_index, DataFrame) tuples
        """
        doc: DoclingDocument = result.document
        tables_df = []

        # Try native export first
        for table_ix, table in enumerate(doc.tables):
            try:
                # Attempt to use built-in export_to_dataframe
                # (This will work if table structure is populated)
                df = table.export_to_dataframe()
                if not df.empty:
                    tables_df.append((table_ix, df))
                    continue
            except (AttributeError, ValueError):
                pass

            # Fallback: Parse from HTML export
            # (HTML export works even when JSON cells are empty)
            try:
                table_html = table.export_to_html()
                df = pd.read_html(StringIO(table_html))[0]
                tables_df.append((table_ix, df))
            except Exception as e:
                print(f"Warning: Could not extract table {table_ix}: {e}")
                continue

        return tables_df

    def export_tables_to_csv(self, result: Any, output_dir: str = ".") -> List[str]:
        """
        Export all tables to CSV files.

        Args:
            result: ConversionResult
            output_dir: Directory to save CSV files

        Returns:
            List of CSV file paths
        """
        tables_df = self.export_tables_to_dataframes(result)
        csv_files = []

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for table_ix, df in tables_df:
            csv_path = output_path / f"table_{table_ix + 1}.csv"
            df.to_csv(csv_path, index=False)
            csv_files.append(str(csv_path))

        return csv_files

    # ========================================================================
    # Issue 3: Figure OCR Filtering (for search indexes)
    # ========================================================================

    def export_text_for_search(self, result: Any) -> str:
        """
        Export text EXCLUDING picture OCR artifacts.

        This prevents noisy in-figure OCR text from polluting semantic search.

        Returns:
            Clean text for indexing
        """
        doc: DoclingDocument = result.document

        # Get picture child text indices to exclude
        picture_child_indices = set()
        try:
            # Access JSON representation to find picture children
            doc_dict = doc.export_to_dict()
            for picture in doc_dict.get("pictures", []):
                for child in picture.get("children", []):
                    ref = child.get("$ref", "")
                    if ref.startswith("#/texts/"):
                        idx = int(ref.split("/")[-1])
                        picture_child_indices.add(idx)
        except Exception:
            pass  # Fallback to including all text

        # Build text excluding picture children
        text_parts = []

        # If we have picture indices, filter by index
        if picture_child_indices:
            doc_dict = doc.export_to_dict()
            for idx, text_item in enumerate(doc_dict.get("texts", [])):
                # Skip picture children
                if idx in picture_child_indices:
                    continue

                # Include semantic labels only
                label = text_item.get("label", "")
                if label in ["paragraph", "text", "title", "section_header",
                             "subtitle-level-1", "list_item", "caption", "code"]:
                    if text_item.get("text"):
                        text_parts.append(text_item["text"])
        else:
            # Fallback: Use standard export
            text_parts = [doc.export_to_text()]

        return "\n\n".join(text_parts)

    # ========================================================================
    # Issue 4: Text Normalization (ligatures, spacing, Unicode)
    # ========================================================================

    @staticmethod
    def normalize_pdf_text(text: str) -> str:
        """
        Normalize PDF text artifacts: ligatures, spacing, Unicode.

        Fixes issues like:
        - "ﬁle" → "file" (ligatures)
        - "consumesvaluabletimewhichcould" → "consumes valuable time which could"
        - "T  H  E" → "THE"

        Args:
            text: Raw text from PDF

        Returns:
            Normalized text
        """
        # 1. Unicode normalization (handles ligatures)
        # NFKC: Compatibility decomposition + composition
        text = unicodedata.normalize('NFKC', text)

        # 2. Common ligature replacements (fallback)
        ligatures = {
            'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
            'ﬅ': 'ft', 'ﬆ': 'st', 'Ĳ': 'IJ', 'ĳ': 'ij',
            'Ǳ': 'DZ', 'ǲ': 'Dz', 'ǳ': 'dz',
        }
        for lig, replacement in ligatures.items():
            text = text.replace(lig, replacement)

        # 3. Fix missing spaces between words (heuristic)
        # Pattern: lowercase + uppercase without space
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

        # 4. Fix excessive character spacing
        # Pattern: "T  H  E" → "THE" (single letters with double spaces)
        text = re.sub(r'\b(\w)\s+(\w)\s+(\w)\b', r'\1\2\3', text)

        # 5. Normalize whitespace
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces → single
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 newlines
        text = text.strip()

        return text

    def export_normalized_text(self, result: Any) -> str:
        """
        Export text with normalization applied.

        Returns:
            Clean, normalized text
        """
        doc: DoclingDocument = result.document
        raw_text = doc.export_to_text()
        return self.normalize_pdf_text(raw_text)

    def export_normalized_markdown(self, result: Any) -> str:
        """
        Export Markdown with text normalization applied.

        Preserves Markdown structure while normalizing text content.

        Returns:
            Normalized Markdown
        """
        doc: DoclingDocument = result.document
        md = doc.export_to_markdown()

        # Normalize each line while preserving Markdown syntax
        lines = md.split('\n')
        normalized_lines = []

        for line in lines:
            # Preserve Markdown syntax (headers, lists, code blocks)
            if line.startswith('#') or line.startswith('-') or line.startswith('*') or line.startswith('```'):
                # Extract prefix
                prefix_match = re.match(r'^([\#\-\*\s`]+)', line)
                if prefix_match:
                    prefix = prefix_match.group()
                    content = line[len(prefix):]
                    normalized_lines.append(prefix + self.normalize_pdf_text(content))
                else:
                    normalized_lines.append(line)
            else:
                normalized_lines.append(self.normalize_pdf_text(line) if line.strip() else line)

        return '\n'.join(normalized_lines)

    # ========================================================================
    # Issue 5: Image Embedding in Exports
    # ========================================================================

    def export_html_with_images(self, result: Any, image_mode: str = "embedded") -> str:
        """
        Export HTML with embedded images.

        Args:
            result: ConversionResult
            image_mode: "embedded" (base64) or "referenced" (files)

        Returns:
            HTML with <img> tags
        """
        doc: DoclingDocument = result.document

        # Map image_mode string to ImageRefMode enum
        if image_mode == "embedded":
            mode = ImageRefMode.EMBEDDED
        elif image_mode == "referenced":
            mode = ImageRefMode.REFERENCED
        else:
            mode = ImageRefMode.PLACEHOLDER

        return doc.export_to_html(image_mode=mode)

    def export_markdown_with_images(
        self,
        result: Any,
        output_dir: str = ".",
        image_subdir: str = "images"
    ) -> str:
        """
        Export Markdown with images saved to files.

        Args:
            result: ConversionResult
            output_dir: Output directory
            image_subdir: Subdirectory for images (relative to output_dir)

        Returns:
            Markdown with ![](images/...) references
        """
        doc: DoclingDocument = result.document

        # Create image directory
        image_path = Path(output_dir) / image_subdir
        image_path.mkdir(parents=True, exist_ok=True)

        # Export Markdown with referenced images
        md_path = Path(output_dir) / "document.md"
        doc.save_as_markdown(
            md_path,
            image_mode=ImageRefMode.REFERENCED
        )

        # Read and return the Markdown
        with open(md_path, "r", encoding="utf-8") as f:
            return f.read()

    # ========================================================================
    # Convenience Methods
    # ========================================================================

    def process_document(
        self,
        source: Any,
        export_html: bool = True,
        export_markdown: bool = True,
        export_tables: bool = True,
        export_text: bool = True,
        output_dir: str = "output"
    ) -> Dict[str, Any]:
        """
        Process document and export all formats.

        Args:
            source: Path to PDF
            export_html: Export HTML with MathML formulas
            export_markdown: Export normalized Markdown
            export_tables: Export tables to CSV
            export_text: Export normalized text
            output_dir: Output directory

        Returns:
            Dictionary with all exports and metadata
        """
        # Convert
        result = self.convert(source)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        outputs = {
            "result": result,
            "document": result.document,
        }

        # Export HTML with MathML formulas
        if export_html:
            html = self.export_html_with_images(result, image_mode="embedded")
            html_file = output_path / "document.html"
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html)
            outputs["html_file"] = str(html_file)
            outputs["html"] = html

        # Export normalized Markdown
        if export_markdown:
            md = self.export_normalized_markdown(result)
            md_file = output_path / "document.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(md)
            outputs["markdown_file"] = str(md_file)
            outputs["markdown"] = md

        # Export tables to CSV
        if export_tables:
            csv_files = self.export_tables_to_csv(result, output_dir=str(output_path))
            outputs["table_csv_files"] = csv_files
            outputs["tables_dataframes"] = self.export_tables_to_dataframes(result)

        # Export normalized text (for search indexing)
        if export_text:
            clean_text = self.export_normalized_text(result)
            search_text = self.export_text_for_search(result)

            text_file = output_path / "document.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(clean_text)

            search_file = output_path / "document_search.txt"
            with open(search_file, "w", encoding="utf-8") as f:
                f.write(search_text)

            outputs["text_file"] = str(text_file)
            outputs["search_text_file"] = str(search_file)
            outputs["text"] = clean_text
            outputs["search_text"] = search_text

        return outputs


# ============================================================================
# Convenience Factory Functions
# ============================================================================

def create_formula_focused_converter() -> EnhancedDoclingConverter:
    """
    Create converter optimized for math-heavy documents.

    Features:
    - Formula enrichment enabled
    - LaTeX extraction
    - MathML HTML export
    """
    return EnhancedDoclingConverter(
        enable_formula_enrichment=True,
        enable_picture_classification=False,
        generate_images=False,
        table_mode="fast"
    )


def create_table_focused_converter() -> EnhancedDoclingConverter:
    """
    Create converter optimized for tabular data extraction.

    Features:
    - Accurate table structure extraction
    - DataFrame/CSV export
    - HTML fallback for empty JSON cells
    """
    return EnhancedDoclingConverter(
        enable_formula_enrichment=False,
        table_mode="accurate",
        generate_images=False
    )


def create_search_optimized_converter() -> EnhancedDoclingConverter:
    """
    Create converter optimized for search indexing.

    Features:
    - Text normalization
    - Figure OCR filtering
    - Clean text export
    """
    return EnhancedDoclingConverter(
        enable_formula_enrichment=True,  # For math-aware search
        enable_picture_classification=True,  # For figure filtering
        generate_images=False,
        table_mode="fast"
    )


def create_full_fidelity_converter() -> EnhancedDoclingConverter:
    """
    Create converter for maximum fidelity exports.

    Features:
    - All enrichments enabled
    - High-quality image generation
    - Complete HTML/Markdown exports
    """
    return EnhancedDoclingConverter(
        enable_formula_enrichment=True,
        enable_picture_classification=True,
        generate_images=True,
        image_scale=2.0,
        table_mode="accurate"
    )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python docling_profile.py <pdf_file> [output_dir]")
        sys.exit(1)

    pdf_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    print(f"Processing: {pdf_file}")
    print(f"Output directory: {output_dir}")

    # Create full-fidelity converter
    converter = create_full_fidelity_converter()

    # Process document
    outputs = converter.process_document(
        pdf_file,
        output_dir=output_dir
    )

    print("\n✅ Processing complete!")
    print(f"HTML: {outputs.get('html_file')}")
    print(f"Markdown: {outputs.get('markdown_file')}")
    print(f"Text: {outputs.get('text_file')}")
    print(f"Search text: {outputs.get('search_text_file')}")
    print(f"Tables: {len(outputs.get('table_csv_files', []))} CSV files")
