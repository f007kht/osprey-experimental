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

### Restart Procedure

Knowing when and how to restart the application is essential for applying changes and troubleshooting issues.

#### When to Restart

You should restart the application in the following scenarios:

- **After modifying code**: Any changes to `app.py` or other Python files require a restart to take effect
- **After changing environment variables**: Both locally and on Hugging Face Spaces, environment variable changes only apply after restart
- **After installing or updating dependencies**: New packages from `requirements.txt` won't be available until restart
- **When the app becomes unresponsive**: If the app freezes, crashes, or shows persistent errors, restart may resolve issues
- **After configuration changes**: Changes to `.streamlit/config.toml` require a restart
- **When Streamlit cache needs clearing**: Cache corruption or stale data may require restart after clearing cache

#### Local Development Restart

To restart the Streamlit app during local development:

1. **Stop the current process:**
   - Press `Ctrl+C` in the terminal where Streamlit is running
   - Or terminate the process using Task Manager (Windows) / Activity Monitor (Mac)

2. **Optional - Clear Streamlit cache** (if experiencing cache-related issues):
   ```bash
   # Windows PowerShell
   Remove-Item -Path "$env:USERPROFILE\.streamlit\cache" -Recurse -Force
   
   # Linux/Mac
   rm -rf ~/.streamlit/cache
   ```

3. **Restart the app:**
   ```bash
   # Standard method
   streamlit run app.py
   
   # Or use the startup script (Windows PowerShell)
   .\start_streamlit.ps1
   ```

4. **Verify the restart:**
   - Check the terminal for startup messages (should show "You can now view your Streamlit app in your browser")
   - Open or refresh your browser at `http://localhost:8501`
   - Verify the app loads correctly and any code changes are reflected

**Note:** If you see "Address already in use" errors, ensure all previous Streamlit processes are terminated before restarting.

#### Hugging Face Spaces Restart

To restart the application on Hugging Face Spaces:

1. **Via UI (Manual Restart):**
   - Navigate to your Space: https://huggingface.co/spaces/f007kht/osprey-experimental
   - Click the **Settings** tab (gear icon in the top right)
   - Scroll to the bottom and click **Restart this Space**
   - Wait for the Space to rebuild (typically 1-3 minutes)

2. **Automatic Restart:**
   - Occurs automatically after code is pushed to the `main` branch on GitHub
   - The GitHub Actions workflow syncs changes and triggers a rebuild
   - No manual intervention needed for code updates

3. **After Environment Variable Changes:**
   - **Always restart** after adding or modifying environment variables
   - Changes in the Settings → Variables and secrets section require restart to apply
   - Verify changes by checking the app sidebar or logs after restart

4. **Verify the restart:**
   - Check the Space **Logs** tab for startup messages
   - Verify the app loads correctly at the Space URL
   - Confirm environment variable changes are reflected (e.g., MongoDB configuration appears in sidebar)

**Important:** Hugging Face Spaces may take 1-3 minutes to fully restart and rebuild. Check the logs to ensure the restart completed successfully.

#### Troubleshooting Restart Issues

**Port Already in Use (Local Development):**
```bash
# Windows PowerShell - Kill existing Streamlit processes
Get-Process python,streamlit -ErrorAction SilentlyContinue | Stop-Process -Force

# Linux/Mac - Kill processes on port 8501
lsof -ti:8501 | xargs kill -9
```

**Cache Issues:**
- Clear the Streamlit cache directory (see step 2 in Local Development Restart above)
- Clear browser cache if UI issues persist
- Restart after clearing cache

**Environment Variables Not Applied:**
- **Local:** Verify environment variables are set correctly in your terminal or `start_streamlit.ps1` script
- **Hugging Face Spaces:** Confirm variables are saved in Settings → Variables and secrets, then restart
- Check that variable names match exactly (case-sensitive)

**App Still Shows Old Behavior After Restart:**
- Hard refresh browser: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- Clear browser cache completely
- Try incognito/private browsing mode to rule out browser cache issues

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
   - **Restart your Space:** Go to **Settings** → **Restart this Space** (see [Restart Procedure](#restart-procedure) section for detailed instructions)

**Important Security Notes:**
- Use **Secrets** (not Variables) for sensitive data like MongoDB connection strings and API keys
- Secrets are hidden in logs but visible in Space settings UI
- Never commit secrets to GitHub - use Hugging Face Spaces secrets instead

**Verification:**
After restarting, the sidebar should show MongoDB configuration options instead of "MongoDB features are disabled" message.

### Viewing Documents with MongoDB Compass

MongoDB Compass is a GUI tool that makes it easy to view, query, and analyze your stored documents. It's particularly useful for:

- 🎨 **Better visualization** - See documents in a user-friendly interface
- 🔎 **Advanced filtering** - Query documents with complex filters
- 📊 **Schema analysis** - Understand your data structure
- 📈 **Query performance insights** - Optimize your queries

#### Step 1: Install MongoDB Compass

1. Download MongoDB Compass from: https://www.mongodb.com/try/download/compass
2. Install the application (available for Windows, macOS, and Linux)

#### Step 2: Connect to MongoDB Atlas

1. Open MongoDB Compass
2. Click **"New Connection"**
3. Paste your connection string:
   ```
   mongodb+srv://<username>:<password>@cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
   - Replace `<username>` and `<password>` with your MongoDB Atlas credentials
   - Replace `cluster.xxxxx.mongodb.net` with your actual cluster address
4. Click **"Connect"**

#### Step 3: Browse Your Data

Once connected:

1. **Expand your database** - You'll see a list of databases in the left sidebar
2. **Click on your collection** - Navigate to your documents collection (default: `documents`)
3. **View documents** - See all your stored documents in a nice GUI with:
   - Document viewer with syntax highlighting
   - Search and filter capabilities
   - Export functionality for analysis

**Tip:** The vector search index is created automatically when documents are saved. It may take a few minutes to be ready for queries. You can verify index creation in the MongoDB Atlas UI under "Search" → "Vector Search Indexes".

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
- **App not reflecting changes:** If code changes aren't appearing, try restarting the app (see [Restart Procedure](#restart-procedure))

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
