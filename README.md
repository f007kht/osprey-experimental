---
title: Docling Document Processor
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
sdk_version: 1.51.0
app_port: 8501
---

# Osprey Docling Document Processor

A Streamlit web application for processing documents using the Docling library. Upload documents (PDF, DOCX, PPTX, XLSX, images, audio) and extract their content as Markdown.

## Features

- 📄 Document upload via web interface
- 🔄 Automatic document processing with Docling
- 📝 Markdown output display
- 🎯 Support for multiple formats: PDF, DOCX, PPTX, XLSX, HTML, images (PNG, JPG, TIFF), and audio (WAV, MP3)

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

3. **Open your browser:**
   The app will automatically open at `http://localhost:8501`

### Deployment

The app is deployed on **Hugging Face Spaces**: https://huggingface.co/spaces/f007kht/osprey-experimental

**Key deployment notes:**
- Uses Docker + Streamlit on Hugging Face Spaces
- 16GB RAM available (vs 1GB on Streamlit Cloud free tier)
- Uses `requirements.txt` for dependency management
- Configured with CPU-only PyTorch builds for cloud compatibility
- System dependencies (Tesseract OCR, libGL) installed via Dockerfile
- **Explicitly configured for CPU-only operation** - prevents GPU detection issues that can crash Streamlit

**Previous deployment attempts on Streamlit Cloud:**
The deployment process and troubleshooting steps for Streamlit Cloud are documented in `deployment-logs/streamlit-cloud-deployment-log.txt`. Memory limitations (1GB RAM) on Streamlit Cloud's free tier led to migration to Hugging Face Spaces.

**Why Hugging Face Spaces:**

The project was moved to Hugging Face Spaces for three primary reasons:

1. **File Size Limitations**: Docling's required model files and dependencies exceed Streamlit Cloud's deployment specifications. Hugging Face Spaces supports Docker deployments with larger storage capacities, allowing all necessary models and dependencies to be included.

2. **GPU/Compute Detection Issues**: The platform provides reliable CPU-only compute environments, and the app is explicitly configured to use CPU mode (`AcceleratorDevice.CPU`) to ensure stable operation without GPU driver dependencies or detection failures.

3. **Memory Resources**: Hugging Face Spaces provides 16GB RAM (vs 1GB on Streamlit Cloud free tier), which is essential for processing large documents with Docling's ML models.

### Configuring Environment Variables

To enable MongoDB storage and other optional features in your Hugging Face Space deployment, you need to configure environment variables:

1. **Navigate to your Space settings:**
   - Go to https://huggingface.co/spaces/f007kht/osprey-experimental
   - Click the **Settings** tab (gear icon in the top right)

2. **Add environment variables:**
   - Scroll to **Variables and secrets** section
   - Click **New variable** or **New secret** (use secrets for sensitive data like connection strings)

3. **Required variables for MongoDB:**

   | Variable Name | Value | Description |
   |--------------|-------|-------------|
   | `ENABLE_MONGODB` | `true` | Enables MongoDB storage features |
   | `MONGODB_CONNECTION_STRING` | `mongodb+srv://...` | Your MongoDB Atlas connection string (mark as secret) |

4. **Optional variables:**

   | Variable Name | Default | Description |
   |--------------|---------|-------------|
   | `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model for RAG |
   | `ENABLE_DOWNLOADS` | `true` | Enable download buttons for processed documents |
   | `USE_REMOTE_EMBEDDINGS` | `false` | Use VoyageAI instead of local embeddings |
   | `VOYAGEAI_API_KEY` | - | Required if `USE_REMOTE_EMBEDDINGS=true` |

5. **After adding variables:**
   - Click **Save** to apply changes
   - Restart your Space: Go to **Settings** → **Restart this Space**

**Important Security Notes:**
- Use **Secrets** (not Variables) for sensitive data like MongoDB connection strings and API keys
- Secrets are hidden in logs but visible in Space settings UI
- Never commit secrets to GitHub - use Hugging Face Spaces secrets instead

**Verification:**
After restarting, the sidebar should show MongoDB configuration options instead of "MongoDB features are disabled" message.

## Project Structure

```
.
├── app.py                              # Main Streamlit application
├── Dockerfile                          # Docker configuration for HF Spaces
├── requirements.txt                    # Python dependencies
├── packages.txt                        # System dependencies (for reference)
├── deployment-logs/                    # Deployment logs and troubleshooting
│   └── streamlit-cloud-deployment-log.txt
└── README.md                           # This file
```

## Requirements

- Python 3.9+
- Streamlit 1.28.0+
- Docling 2.60.0+
- PyTorch (CPU-only version for cloud deployment)

## Troubleshooting

### Deployment Issues

If you encounter deployment issues on Streamlit Cloud, refer to `deployment-logs/streamlit-cloud-deployment-log.txt` for the complete deployment history and resolution steps.

**Common fixes applied:**
1. Removed `pyproject.toml` with TOML parse errors
2. Removed `uv.lock` to force use of `requirements.txt`
3. Configured CPU-only PyTorch installation for cloud compatibility

### Local Issues

- **First run is slow:** Docling downloads AI models on first use, which may take several minutes
- **Memory requirements:** Processing large documents may require significant memory
- **File upload limits:** Check Streamlit Cloud limits for file upload sizes

### Known Issues and Fixes

**Issue: `DoclingPdfParser.load() got an unexpected keyword argument 'password'`**

- **Symptoms:** Error occurs when processing PDF documents (especially password-protected ones)
- **Cause:** Version mismatch between Docling code and installed `docling-parse` library - some versions don't support the `password` parameter in `DoclingPdfParser.load()`
- **Fix Applied:** Modified `docling/backend/docling_parse_v4_backend.py` to handle API compatibility:
  - Tries loading with password parameter first (for newer versions)
  - Falls back to loading without password if TypeError occurs (since pypdfium2 already handles password protection)
- **Status:** Fixed in commit 97f3fb0

**Issue: Streamlit fails to start due to GPU/compute detection issues**

- **Symptoms:** Streamlit app doesn't start or crashes during initialization on Hugging Face Spaces
- **Cause:** Docling defaults to `AcceleratorDevice.AUTO` which attempts GPU/CUDA/MPS detection, causing issues on CPU-only platforms like Hugging Face Spaces
- **Fix Applied:** Explicitly set `pipeline_options.accelerator_options.device = AcceleratorDevice.CPU` in `app.py`
- **Why:** This was a key reason for moving to Hugging Face Spaces - the platform provides CPU-only compute, and forcing CPU mode ensures reliable operation without GPU driver dependencies
- **Status:** Fixed - app now explicitly uses CPU device

## Remote Synchronization

This project maintains synchronized repositories on GitHub and Hugging Face Spaces:

- **GitHub Repository**: https://github.com/f007kht/osprey-experimental (source of truth)
- **Hugging Face Space**: https://huggingface.co/spaces/f007kht/osprey-experimental (automatically synced)

### Synchronization Strategy

- **Single Source of Truth**: GitHub (`origin/main`) is the authoritative source for all code changes
- **Automated Sync**: A GitHub Actions workflow (`.github/workflows/sync-to-hf.yml`) automatically syncs the `main` branch to Hugging Face Spaces on every push
- **No Direct Edits**: Never make direct edits to the Hugging Face Space UI - all changes must flow through GitHub first
- **Prevention of Divergence**: The automated sync ensures both remotes remain perfectly synchronized

### How It Works

1. All development happens on GitHub (feature branches, pull requests, merges to main)
2. When code is pushed to the `main` branch on GitHub, the sync workflow automatically triggers
3. The workflow force-pushes `main` to the Hugging Face Space remote, keeping them in sync
4. Hugging Face Spaces automatically rebuilds the Docker image when changes are detected

### Git LFS Requirement

Large test data files (`.pages.json` files > 10MB) are tracked using Git LFS. This is required because Hugging Face Spaces has a 10MB file size limit without LFS. The `.gitattributes` file configures which files use LFS.

### Manual Sync (if needed)

If you need to manually sync (not recommended, as automated sync should handle this):

```bash
git push huggingface main --force
```

**Note**: Manual sync should only be done in emergencies. The automated workflow handles all normal synchronization.

## License

Confidential - Osprey_Intel_LLC
