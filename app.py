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
import tempfile

import streamlit as st

from docling.document_converter import DocumentConverter

# Set up the Streamlit page title
st.title("📄 Docling Document Processor")
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

    st.info(f"Processing `{uploaded_file.name}`... This might take a moment.")

    try:
        # Initialize the converter
        # This will download models on the first run, which can be slow.
        with st.spinner("Initializing Docling (this may take a while on first run)..."):
            converter = DocumentConverter()

        # Run the conversion process
        with st.spinner(f"Converting `{uploaded_file.name}`..."):
            result = converter.convert(tmp_file_path)

        st.success("✅ Document processed successfully!")

        # 3. Display the results
        st.subheader("Extracted Content (Markdown)")

        # Export the document's content to Markdown
        markdown_output = result.document.export_to_markdown()

        # Display the Markdown in the app.
        # We use a text_area for long outputs, but st.markdown() also works.
        st.text_area("Markdown Output", markdown_output, height=600)

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
