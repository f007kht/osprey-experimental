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
# Downloads are always enabled by default (can be disabled via ENABLE_DOWNLOADS=false)
ENABLE_DOWNLOADS = os.getenv("ENABLE_DOWNLOADS", "true").lower() == "true"
# MongoDB storage requires explicit enable via ENABLE_MONGODB=true
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

# Embedding configuration
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

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


def _prepare_download_data(result, format_type: str, base_filename: str = "document") -> tuple[str, str]:
    """
    Prepare data for download in the specified format.
    
    Args:
        result: ConversionResult from Docling
        format_type: One of 'markdown', 'json', 'txt', 'doctags'
        base_filename: Base filename to use for download (default: "document")
    
    Returns:
        tuple of (data_string, filename)
    """
    if format_type == "markdown":
        data = result.document.export_to_markdown()
        filename = _get_download_filename(base_filename, "md")
    elif format_type == "json":
        data = json.dumps(result.document.export_to_dict(), indent=2)
        filename = _get_download_filename(base_filename, "json")
    elif format_type == "txt":
        data = result.document.export_to_markdown(strict_text=True)
        filename = _get_download_filename(base_filename, "txt")
    elif format_type == "doctags":
        data = result.document.export_to_document_tokens()
        filename = _get_download_filename(base_filename, "doctags")
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


def _store_in_mongodb(
    document: Any,
    original_filename: str,
    file_size: int,
    mongodb_connection_string: str,
    database_name: str,
    collection_name: str,
    embedding_config: Optional[Dict[str, Any]] = None
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
        # Connect to MongoDB
        client = MongoClient(mongodb_connection_string)
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
                "embedding_model": model_name,
                "embedding_dimensions": embedding_dim,
                "use_remote": embedding_config.get("use_remote", False)
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
if "use_remote_embeddings" not in st.session_state:
    st.session_state.use_remote_embeddings = False
if "embedding_model_name" not in st.session_state:
    st.session_state.embedding_model_name = DEFAULT_EMBEDDING_MODEL

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
        st.subheader("🔍 Embedding Configuration")
        
        use_remote_embeddings = st.checkbox(
            "Use Remote Embeddings (VoyageAI)",
            value=os.getenv("USE_REMOTE_EMBEDDINGS", "false").lower() == "true",
            help="Use VoyageAI for embeddings (requires API key). Unchecked = local models.",
            disabled=not mongodb_enabled
        )
        
        if use_remote_embeddings:
            voyageai_api_key = st.text_input(
                "VoyageAI API Key",
                value=os.getenv("VOYAGEAI_API_KEY", ""),
                type="password",
                help="API key for VoyageAI embeddings (required if using remote)",
                disabled=not mongodb_enabled
            )
            embedding_model_name = "voyage-context-3"  # Fixed for VoyageAI
        else:
            voyageai_api_key = ""
            embedding_model_name = st.text_input(
                "Local Embedding Model",
                value=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
                help="HuggingFace model name for local embeddings (e.g., sentence-transformers/all-MiniLM-L6-v2)",
                disabled=not mongodb_enabled
            )
        
        if mongodb_enabled and not mongodb_connection_string:
            st.warning("⚠️ MongoDB connection string is required for storage.")
        if mongodb_enabled and use_remote_embeddings and not voyageai_api_key:
            st.warning("⚠️ VoyageAI API key is required when using remote embeddings.")
    else:
        mongodb_enabled = False
        mongodb_connection_string = ""
        mongodb_database = "docling_db"
        mongodb_collection = "documents"
        voyageai_api_key = ""
        use_remote_embeddings = False
        embedding_model_name = DEFAULT_EMBEDDING_MODEL
    
    # Store in session state for use in processing
    st.session_state.mongodb_enabled = mongodb_enabled if ENABLE_MONGODB else False
    st.session_state.mongodb_connection_string = mongodb_connection_string
    st.session_state.mongodb_database = mongodb_database
    st.session_state.mongodb_collection = mongodb_collection
    st.session_state.voyageai_api_key = voyageai_api_key
    st.session_state.use_remote_embeddings = use_remote_embeddings if ENABLE_MONGODB else False
    st.session_state.embedding_model_name = embedding_model_name if ENABLE_MONGODB else DEFAULT_EMBEDDING_MODEL

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

        # 4. Download functionality - enabled by default
        st.divider()
        st.subheader("📥 Download Processed Document")
        
        if ENABLE_DOWNLOADS:
            st.write("Download the processed document in various formats:")
            
            col1, col2, col3, col4 = st.columns(4)
            
            try:
                base_filename = os.path.splitext(uploaded_file.name)[0] if uploaded_file.name else "document"
                
                # Markdown download
                with col1:
                    md_data, md_filename = _prepare_download_data(result, "markdown", base_filename)
                    st.download_button(
                        label="📄 Download Markdown",
                        data=md_data,
                        file_name=md_filename,
                        mime="text/markdown",
                        key=f"download_md_{uploaded_file.name}",
                        use_container_width=True
                    )
                
                # JSON download
                with col2:
                    json_data, json_filename = _prepare_download_data(result, "json", base_filename)
                    st.download_button(
                        label="📋 Download JSON",
                        data=json_data,
                        file_name=json_filename,
                        mime="application/json",
                        key=f"download_json_{uploaded_file.name}",
                        use_container_width=True
                    )
                
                # Plain text download
                with col3:
                    txt_data, txt_filename = _prepare_download_data(result, "txt", base_filename)
                    st.download_button(
                        label="📝 Download Text",
                        data=txt_data,
                        file_name=txt_filename,
                        mime="text/plain",
                        key=f"download_txt_{uploaded_file.name}",
                        use_container_width=True
                    )
                
                # Doctags download
                with col4:
                    doctags_data, doctags_filename = _prepare_download_data(result, "doctags", base_filename)
                    st.download_button(
                        label="🏷️ Download Doctags",
                        data=doctags_data,
                        file_name=doctags_filename,
                        mime="text/plain",
                        key=f"download_doctags_{uploaded_file.name}",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"❌ Download feature unavailable: {e}")
                st.exception(e)
                # App continues normally - download failure doesn't break the app
        else:
            st.warning("⚠️ Downloads are disabled in this deployment.")
            st.info("💡 To enable downloads, set environment variable `ENABLE_DOWNLOADS=true`")

        # 5. MongoDB Storage (if enabled)
        st.divider()
        st.subheader("💾 Database Storage")
        
        if ENABLE_MONGODB:
            if st.session_state.get("mongodb_enabled", False):
                # Check if MongoDB is properly configured
                use_remote = st.session_state.get("use_remote_embeddings", False)
                has_api_key = bool(st.session_state.get("voyageai_api_key", ""))
                is_configured = st.session_state.mongodb_connection_string and (
                    (use_remote and has_api_key) or (not use_remote)
                )
                
                if is_configured:
                    col_save, col_info = st.columns([1, 2])
                    with col_save:
                        if st.button("💾 Save to MongoDB", key=f"save_mongodb_{uploaded_file.name}", use_container_width=True):
                            try:
                                with st.spinner("Saving document to MongoDB with embeddings..."):
                                    embedding_config = {
                                        "use_remote": st.session_state.use_remote_embeddings,
                                        "model_name": st.session_state.embedding_model_name
                                    }
                                    if st.session_state.use_remote_embeddings:
                                        embedding_config["api_key"] = st.session_state.voyageai_api_key
                                    
                                    success = _store_in_mongodb(
                                        document=result.document,
                                        original_filename=uploaded_file.name,
                                        file_size=len(uploaded_file.getvalue()),
                                        mongodb_connection_string=st.session_state.mongodb_connection_string,
                                        database_name=st.session_state.mongodb_database,
                                        collection_name=st.session_state.mongodb_collection,
                                        embedding_config=embedding_config
                                    )
                                    
                                    if success:
                                        st.success("✅ Document saved to MongoDB successfully!")
                                        st.info("💡 The vector search index is being created automatically. It may take a few minutes to be ready for queries.")
                                    else:
                                        st.error("❌ Failed to save document to MongoDB. Check error messages above.")
                            except ImportError as e:
                                st.error(f"❌ Missing dependency: {e}")
                                if st.session_state.use_remote_embeddings:
                                    st.info("💡 Install MongoDB dependencies with: pip install pymongo[srv] voyageai")
                                else:
                                    st.info("💡 Install MongoDB dependencies with: pip install pymongo[srv] sentence-transformers")
                            except ValueError as e:
                                st.error(f"❌ Configuration error: {e}")
                            except Exception as e:
                                st.error(f"❌ Error saving to MongoDB: {e}")
                                st.exception(e)
                                # App continues normally - storage failure doesn't break the app
                    with col_info:
                        st.info("💡 Stored documents include full markdown, JSON, and vector embeddings for RAG search.")
                else:
                    st.warning("⚠️ MongoDB storage is enabled but not configured.")
                    if use_remote:
                        st.info("ℹ️ Configure MongoDB connection and VoyageAI API key in the sidebar (⚙️ Configuration) to enable storage.")
                    else:
                        st.info("ℹ️ Configure MongoDB connection in the sidebar (⚙️ Configuration) to enable storage.")
            else:
                st.info("ℹ️ MongoDB storage is available but not enabled.")
                st.info("💡 Enable it in the sidebar under '⚙️ Configuration' → '📊 MongoDB Storage'")
                if not PYMONGO_AVAILABLE:
                    st.warning("⚠️ MongoDB dependencies not installed. Install with: `pip install pymongo[srv] sentence-transformers`")
        else:
            st.info("ℹ️ MongoDB storage feature is not enabled in this deployment.")
            st.info("💡 To enable: Set environment variable `ENABLE_MONGODB=true` and install dependencies: `pymongo[srv] sentence-transformers`")

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
