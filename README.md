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

The app is deployed on **Hugging Face Spaces**: https://huggingface.co/spaces/f007kht/osprey-docling

**Key deployment notes:**
- Uses Docker + Streamlit on Hugging Face Spaces
- 16GB RAM available (vs 1GB on Streamlit Cloud free tier)
- Uses `requirements.txt` for dependency management
- Configured with CPU-only PyTorch builds for cloud compatibility
- System dependencies (Tesseract OCR, libGL) installed via Dockerfile

**Previous deployment attempts on Streamlit Cloud:**
The deployment process and troubleshooting steps for Streamlit Cloud are documented in `deployment-logs/streamlit-cloud-deployment-log.txt`. Memory limitations (1GB RAM) on Streamlit Cloud's free tier led to migration to Hugging Face Spaces.

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

## License

Confidential - Osprey_Intel_LLC
