# Application Startup Protocol

This document explains how to start the Docling Document Processor application in different environments.

## What It Does (Plain English)

### Local Development (Streamlit)

Run `streamlit run app.py`, or on Windows run `start_streamlit.ps1` which sets up environment variables (enables Mongo, sets connection string, forces UTF-8).

### Docker / Hugging Face Spaces

The image's entrypoint sets up Tesseract (`TESSDATA_PREFIX`) and then runs Streamlit bound to `0.0.0.0` using the platform-provided `$PORT` environment variable (defaults to 8501).

### App Boot Sequence (inside `app.py`)

1. **Clears `DOCLING_ARTIFACTS_PATH`** so Docling doesn't reuse a bad cache.
2. **Disables OpenCV's EXR** to avoid headless GPU/libGL crashes.
3. **Auto-detects `TESSDATA_PREFIX`** if you didn't set it (with graceful degradation).
4. **Reads feature flags** (`ENABLE_DOWNLOADS`, `ENABLE_MONGODB`).
5. **Memory throttle**: `PAGE_BATCH_SIZE` env var (default: 1, safe but slower; higher = faster but more memory).
6. **Forces CPU device** (avoids flaky GPU detection on clouds).
7. **Validates MongoDB connection** at startup if enabled (non-blocking, logs status).
8. **Health check endpoint**: Access via `?health=1` query parameter.

### Ports

- **Local**: `8501` (default Streamlit port)
- **Docker/Spaces**: Uses `$PORT` environment variable if set, otherwise defaults to `8501`

### Environment Variables

**Required for MongoDB:**
- `ENABLE_MONGODB=true` - Enable MongoDB features
- `MONGODB_CONNECTION_STRING` - MongoDB Atlas connection string (e.g., `mongodb+srv://...`)

**Optional Configuration:**
- `PAGE_BATCH_SIZE` - Docling page batch size (default: 1, range: 1-10)
- `EMBEDDING_MODEL` - Local embedding model (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `USE_REMOTE_EMBEDDINGS` - Use VoyageAI instead of local embeddings (default: `false`)
- `VOYAGEAI_API_KEY` - Required if `USE_REMOTE_EMBEDDINGS=true`
- `MONGODB_DATABASE` - Database name (default: `docling_documents`)
- `MONGODB_COLLECTION` - Collection name (default: `documents`)
- `TESSDATA_PREFIX` - Tesseract OCR data directory (auto-detected if not set)
- `PORT` - Server port (default: 8501, used by platforms like Spaces, Render, Fly, Heroku)

### How to Verify

- Streamlit banner appears in terminal
- Browser opens automatically to `http://localhost:8501`
- First run may download models (takes several minutes)
- Check logs for MongoDB connection status: `MongoDB: connected to <db>/<collection>` or `MongoDB: disabled`
- Health check: Visit `http://localhost:8501?health=1` for JSON status

## Golden Path Snippets

### Windows (Dev)

```powershell
# start_streamlit.ps1 (core)
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
$env:LC_ALL="C.UTF-8"

# Set only if you actually want DB writes
$env:ENABLE_MONGODB="true"
$env:MONGODB_CONNECTION_STRING="<your-connection-string>"

streamlit run app.py
```

Or simply run:
```powershell
.\start_streamlit.ps1
```

### Linux/Mac (Dev)

```bash
export PYTHONIOENCODING="utf-8"
export PYTHONUTF8="1"
export LC_ALL="C.UTF-8"

# Optional: Enable MongoDB
export ENABLE_MONGODB="true"
export MONGODB_CONNECTION_STRING="<your-connection-string>"

streamlit run app.py
```

### Docker Entrypoint

The `entrypoint.sh` script (used by Dockerfile):

```bash
#!/usr/bin/env bash
set -euo pipefail

export TESSDATA_PREFIX="${TESSDATA_PREFIX:-/usr/share/tesseract-ocr/4.00/tessdata}"

exec streamlit run app.py --server.address=0.0.0.0 --server.port="${PORT:-8501}"
```

## Startup Improvements

### 1. Platform Port Support

The application now uses `$PORT` environment variable when available (for Hugging Face Spaces, Render, Fly, Heroku, etc.), falling back to 8501 if not set.

### 2. Centralized Settings

All environment variables are managed through `app_settings.py` using `pydantic-settings`, providing:
- Single source of truth for configuration
- Validation and sensible defaults
- Optional `.env` file support (via `python-dotenv`)

### 3. Graceful OCR Degradation

If `TESSDATA_PREFIX` cannot be resolved:
- Logs a warning (does not crash)
- Falls back to CLI OCR mode (auto-detects language data)
- App continues to function normally

### 4. Configurable Memory Tuning

`PAGE_BATCH_SIZE` environment variable controls memory vs. performance tradeoff:
- `1` (default): Safe, memory-efficient, slower
- `2-4`: Balanced
- `5-10`: Faster but uses more memory

### 5. MongoDB Boot Validation

When `ENABLE_MONGODB=true`:
- Validates connection string format at startup
- Tests connection with quick ping (non-blocking, short timeout)
- Logs one-liner: `MongoDB: connected to <db>/<collection>` or error message
- Ensures TLS/SSL options are properly configured

### 6. Health Check Endpoint

Access via query parameter: `?health=1`

Returns JSON:
```json
{
  "status": "ok",
  "mongodb": "connected|disabled|error",
  "ocr": "available|degraded|unavailable",
  "page_batch_size": 1,
  "port": 8501,
  "mongodb_db_collection": "docling_documents/documents"  // if connected
}
```

## Sanity Checklist Before You Run

- ✅ `TESSDATA_PREFIX` resolves (or the app auto-detects it)
- ✅ If using Mongo: `ENABLE_MONGODB=true` + a working `MONGODB_CONNECTION_STRING` set in your environment/Secrets
- ✅ On Spaces/Docker: app listens on `${PORT}` and `0.0.0.0`
- ✅ No GPU assumptions (CPU-only is intentional)
- ✅ Secrets are NOT committed to version control (use environment variables or `.env` file excluded via `.gitignore`)

## Troubleshooting

### OCR Not Available

If you see `OCR_STATUS: degraded`:
- Set `TESSDATA_PREFIX` environment variable to your Tesseract data directory
- On Linux: Usually `/usr/share/tesseract-ocr/4.00/tessdata/` or `/usr/share/tesseract-ocr/5/tessdata/`
- App will still work using CLI OCR mode (slower but functional)

### MongoDB Connection Fails

Check the logs for specific error messages:
- `MongoDB: configuration error - ...` - Invalid connection string format
- `MongoDB: connection failed - ...` - Network/authentication issue
- Verify IP whitelist in MongoDB Atlas (for cloud deployments, use `0.0.0.0/0`)
- Verify connection string format: `mongodb+srv://...` or `mongodb://...`

### Port Already in Use

```bash
# Windows PowerShell
Get-Process python,streamlit -ErrorAction SilentlyContinue | Stop-Process -Force

# Linux/Mac
lsof -ti:8501 | xargs kill -9
```

### Settings Not Loading

- Verify `pydantic-settings` and `python-dotenv` are installed: `pip install -r requirements.txt`
- Check that environment variables are set correctly (use `echo $VARIABLE_NAME` on Linux/Mac or `$env:VARIABLE_NAME` on PowerShell)
- If using `.env` file, ensure it's in the project root directory

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_DOWNLOADS` | `true` | Enable download buttons for processed documents |
| `ENABLE_MONGODB` | `false` | Enable MongoDB storage features |
| `MONGODB_CONNECTION_STRING` | - | MongoDB Atlas connection string (required if `ENABLE_MONGODB=true`) |
| `MONGODB_DATABASE` | `docling_documents` | MongoDB database name |
| `MONGODB_COLLECTION` | `documents` | MongoDB collection name |
| `PAGE_BATCH_SIZE` | `1` | Docling page batch size (1-10, higher = faster but more memory) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model for RAG |
| `USE_REMOTE_EMBEDDINGS` | `false` | Use VoyageAI instead of local embeddings |
| `VOYAGEAI_API_KEY` | - | VoyageAI API key (required if `USE_REMOTE_EMBEDDINGS=true`) |
| `PORT` | `8501` | Server port (use `$PORT` on platforms like Spaces) |
| `TESSDATA_PREFIX` | Auto-detected | Tesseract OCR data directory path |
| `OPENCV_IO_ENABLE_OPENEXR` | `0` | Disable OpenCV EXR (prevents headless GPU/libGL crashes) |
| `PYTHONIOENCODING` | `utf-8` | Python IO encoding |
| `PYTHONUTF8` | `1` | Enable Python UTF-8 mode |
| `LC_ALL` | `C.UTF-8` | Locale setting |

## See Also

- [README.md](README.md) - Main project documentation
- [app_settings.py](app_settings.py) - Settings module implementation
- [entrypoint.sh](entrypoint.sh) - Docker entrypoint script
- [Dockerfile](Dockerfile) - Docker configuration

