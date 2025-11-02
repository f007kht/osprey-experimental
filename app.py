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
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

# Set environment variable to use headless OpenCV before any imports
# This prevents libGL.so.1 errors on headless systems like Streamlit Cloud
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "0"

# Feature flags - can be controlled via environment variables
ENABLE_DOWNLOADS = os.getenv("ENABLE_DOWNLOADS", "true").lower() == "true"
ENABLE_MONGODB = os.getenv("ENABLE_MONGODB", "false").lower() == "true"

# TESSDATA_PREFIX is now set by the Dockerfile at build time
# The Dockerfile detects the correct tessdata path using dpkg during the build process
# This eliminates the need for runtime detection and ensures a single source of truth

import streamlit as st

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
except Exception as e:
    st.error(f"Failed to import docling modules: {e}")
    st.exception(e)
    st.stop()

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

try:
    import voyageai
    VOYAGEAI_AVAILABLE = True
except ImportError:
    VOYAGEAI_AVAILABLE = False

# Cache the converter to avoid reinitializing on every upload
# This prevents memory spikes from loading models multiple times
@st.cache_resource
def get_converter():
    """Get or create a cached DocumentConverter instance with Tesseract OCR."""
    # Note: Cannot use Streamlit UI elements (st.error, st.exception) inside cached functions
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    # Explicitly set Tesseract OCR to avoid RapidOCR fallback and permission errors
    pipeline_options.ocr_options = TesseractOcrOptions()
    # Force CPU device to avoid GPU/CUDA/MPS detection issues that can prevent Streamlit from starting
    # This ensures the app works on systems without GPU or with GPU driver issues
    pipeline_options.accelerator_options.device = AcceleratorDevice.CPU
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def _get_download_filename(base_filename: str, extension: str) -> str:
    """Generate download filename from original filename."""
    name_without_ext = os.path.splitext(base_filename)[0]
    return f"{name_without_ext}.{extension}"


def _prepare_download_data(result, format_type: str) -> tuple[str, str]:
    """
    Prepare data for download in the specified format.
    
    Args:
        result: ConversionResult from Docling
        format_type: One of 'markdown', 'json', 'txt', 'doctags'
    
    Returns:
        tuple of (data_string, filename)
    """
    base_name = "document"  # Default if filename not available
    
    if format_type == "markdown":
        data = result.document.export_to_markdown()
        filename = _get_download_filename(base_name, "md")
    elif format_type == "json":
        data = json.dumps(result.document.export_to_dict(), indent=2)
        filename = _get_download_filename(base_name, "json")
    elif format_type == "txt":
        data = result.document.export_to_markdown(strict_text=True)
        filename = _get_download_filename(base_name, "txt")
    elif format_type == "doctags":
        data = result.document.export_to_document_tokens()
        filename = _get_download_filename(base_name, "doctags")
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
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


def _generate_embeddings(chunk_texts: List[str], api_key: str) -> List[List[float]]:
    """
    Generate embeddings for chunks using VoyageAI.
    
    Args:
        chunk_texts: List of chunk text strings
        api_key: VoyageAI API key
    
    Returns:
        List of embedding vectors
    """
    if not VOYAGEAI_AVAILABLE:
        raise ImportError("voyageai not available. Install with: pip install voyageai")
    
    if not api_key:
        raise ValueError("VoyageAI API key is required")
    
    vo = voyageai.Client(api_key)
    result = vo.contextualized_embed(
        inputs=[chunk_texts],
        model="voyage-context-3"
    )
    
    # Extract embeddings from result
    embeddings = [emb for r in result.results for emb in r.embeddings]
    return embeddings


def _store_in_mongodb(
    document: Any,
    original_filename: str,
    file_size: int,
    mongodb_connection_string: str,
    database_name: str,
    collection_name: str,
    voyageai_api_key: str
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
        voyageai_api_key: VoyageAI API key for embeddings
    
    Returns:
        True if successful, False otherwise
    """
    if not PYMONGO_AVAILABLE:
        return False
    
    try:
        # Connect to MongoDB
        client = MongoClient(mongodb_connection_string)
        db = client[database_name]
        collection = db[collection_name]
        
        # Chunk the document
        chunks_data = _chunk_document(document)
        chunk_texts = [chunk["text"] for chunk in chunks_data]
        
        # Generate embeddings
        embeddings = _generate_embeddings(chunk_texts, voyageai_api_key)
        
        # Combine chunks with embeddings
        chunks_with_embeddings = []
        for chunk_data, embedding in zip(chunks_data, embeddings):
            chunks_with_embeddings.append({
                "text": chunk_data["text"],
                "embedding": embedding,
                "chunk_index": chunk_data["chunk_index"],
                "metadata": chunk_data["metadata"]
            })
        
        # Prepare document for storage
        doc_to_store = {
            "filename": original_filename,
            "original_filename": original_filename,
            "processed_at": datetime.utcnow().isoformat(),
            "file_size": file_size,
            "docling_json": document.export_to_dict(),
            "markdown": document.export_to_markdown(),
            "chunks": chunks_with_embeddings,
            "metadata": {
                "total_chunks": len(chunks_with_embeddings),
                "embedding_model": "voyage-context-3",
                "embedding_dimensions": len(embeddings[0]) if embeddings else 0
            }
        }
        
        # Insert document
        result = collection.insert_one(doc_to_store)
        
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
                        "numDimensions": 1024,  # voyage-context-3 dimensions
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
        # Don't use st.error here - let the caller handle UI messages
        # This allows the function to be called from different contexts
        raise  # Re-raise to let caller handle it
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

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # MongoDB Configuration (only if enabled)
    if ENABLE_MONGODB:
        st.subheader("📊 MongoDB Storage")
        mongodb_enabled = st.checkbox(
            "Enable MongoDB Storage",
            value=True,
            help="Store processed documents in MongoDB with vector embeddings for RAG"
        )
        
        mongodb_connection_string = st.text_input(
            "MongoDB Connection String",
            value=os.getenv("MONGODB_CONNECTION_STRING", ""),
            type="password",
            help="MongoDB Atlas connection string (e.g., mongodb+srv://...)",
            disabled=not mongodb_enabled
        )
        
        mongodb_database = st.text_input(
            "Database Name",
            value=os.getenv("MONGODB_DATABASE", "docling_db"),
            help="Name of the MongoDB database",
            disabled=not mongodb_enabled
        )
        
        mongodb_collection = st.text_input(
            "Collection Name",
            value=os.getenv("MONGODB_COLLECTION", "documents"),
            help="Name of the MongoDB collection",
            disabled=not mongodb_enabled
        )
        
        st.divider()
        st.subheader("🔑 VoyageAI Configuration")
        voyageai_api_key = st.text_input(
            "VoyageAI API Key",
            value=os.getenv("VOYAGEAI_API_KEY", ""),
            type="password",
            help="API key for VoyageAI embeddings",
            disabled=not mongodb_enabled
        )
        
        if mongodb_enabled and (not mongodb_connection_string or not voyageai_api_key):
            st.warning("⚠️ MongoDB connection string and VoyageAI API key are required for storage.")
    else:
        mongodb_enabled = False
        mongodb_connection_string = ""
        mongodb_database = "docling_db"
        mongodb_collection = "documents"
        voyageai_api_key = ""
    
    # Store in session state for use in processing
    st.session_state.mongodb_enabled = mongodb_enabled if ENABLE_MONGODB else False
    st.session_state.mongodb_connection_string = mongodb_connection_string
    st.session_state.mongodb_database = mongodb_database
    st.session_state.mongodb_collection = mongodb_collection
    st.session_state.voyageai_api_key = voyageai_api_key

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
        # Get the cached converter (initializes once, then reuses)
        # This prevents memory spikes from loading models multiple times
        with st.spinner("Loading Docling (this may take a while on first run)..."):
            converter = get_converter()

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
        # Unique key prevents conflicts when multiple files are processed
        st.text_area("Markdown Output", markdown_output, height=600, key=f"output_{uploaded_file.name}")

        # 4. Download functionality (if enabled)
        if ENABLE_DOWNLOADS:
            st.subheader("📥 Download Processed Document")
            st.write("Download the processed document in various formats:")
            
            col1, col2, col3, col4 = st.columns(4)
            
            try:
                # Markdown download
                with col1:
                    md_data, _ = _prepare_download_data(result, "markdown")
                    st.download_button(
                        label="📄 Markdown",
                        data=md_data,
                        file_name=_get_download_filename(uploaded_file.name, "md"),
                        mime="text/markdown",
                        key=f"download_md_{uploaded_file.name}"
                    )
                
                # JSON download
                with col2:
                    json_data, _ = _prepare_download_data(result, "json")
                    st.download_button(
                        label="📋 JSON",
                        data=json_data,
                        file_name=_get_download_filename(uploaded_file.name, "json"),
                        mime="application/json",
                        key=f"download_json_{uploaded_file.name}"
                    )
                
                # Plain text download
                with col3:
                    txt_data, _ = _prepare_download_data(result, "txt")
                    st.download_button(
                        label="📝 Plain Text",
                        data=txt_data,
                        file_name=_get_download_filename(uploaded_file.name, "txt"),
                        mime="text/plain",
                        key=f"download_txt_{uploaded_file.name}"
                    )
                
                # Doctags download
                with col4:
                    doctags_data, _ = _prepare_download_data(result, "doctags")
                    st.download_button(
                        label="🏷️ Doctags",
                        data=doctags_data,
                        file_name=_get_download_filename(uploaded_file.name, "doctags"),
                        mime="text/plain",
                        key=f"download_doctags_{uploaded_file.name}"
                    )
            except Exception as e:
                st.warning(f"Download feature temporarily unavailable: {e}")
                # App continues normally - download failure doesn't break the app

        # 5. MongoDB Storage (if enabled)
        if ENABLE_MONGODB and st.session_state.get("mongodb_enabled", False):
            st.subheader("💾 MongoDB Storage")
            
            # Check if MongoDB is properly configured
            if (st.session_state.mongodb_connection_string and 
                st.session_state.voyageai_api_key):
                
                if st.button("💾 Save to MongoDB", key=f"save_mongodb_{uploaded_file.name}"):
                    try:
                        with st.spinner("Saving document to MongoDB with embeddings..."):
                            success = _store_in_mongodb(
                                document=result.document,
                                original_filename=uploaded_file.name,
                                file_size=len(uploaded_file.getvalue()),
                                mongodb_connection_string=st.session_state.mongodb_connection_string,
                                database_name=st.session_state.mongodb_database,
                                collection_name=st.session_state.mongodb_collection,
                                voyageai_api_key=st.session_state.voyageai_api_key
                            )
                            
                            if success:
                                st.success("✅ Document saved to MongoDB successfully!")
                                st.info("💡 The vector search index is being created automatically. It may take a few minutes to be ready for queries.")
                            else:
                                st.error("❌ Failed to save document to MongoDB. Check error messages above.")
                    except ImportError as e:
                        st.error(f"❌ Missing dependency: {e}")
                        st.info("💡 Install MongoDB dependencies with: pip install pymongo[srv] voyageai")
                    except ValueError as e:
                        st.error(f"❌ Configuration error: {e}")
                    except Exception as e:
                        st.error(f"❌ Error saving to MongoDB: {e}")
                        st.exception(e)
                        # App continues normally - storage failure doesn't break the app
            else:
                st.info("ℹ️ Configure MongoDB connection and VoyageAI API key in the sidebar to enable storage.")

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
