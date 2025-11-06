# Streamlit startup script for local development
# Sets up environment variables and runs the Streamlit application

# UTF-8 encoding configuration (required for proper text processing)
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
$env:LC_ALL="C.UTF-8"  # More portable than en_US.UTF-8

# MongoDB configuration (set only if you want MongoDB enabled)
# IMPORTANT: Never commit connection strings to version control!
# Set these in your environment or use a .env file (loaded by app_settings.py)
if (-not $env:ENABLE_MONGODB) {
    $env:ENABLE_MONGODB="false"  # Default to disabled
}

# Optional: Set MongoDB connection string if not already set
# Uncomment and set your connection string, or set it as an environment variable:
# $env:MONGODB_CONNECTION_STRING="mongodb+srv://username:password@cluster.mongodb.net/"

# Optional: Configure performance tuning
# $env:PAGE_BATCH_SIZE="1"  # Default: 1 (safe/memory-efficient), higher = faster but more memory

Write-Host "Starting Streamlit application..."
Write-Host "ENABLE_MONGODB=$env:ENABLE_MONGODB"
if ($env:MONGODB_CONNECTION_STRING) {
    Write-Host "MongoDB connection string is set: $($env:MONGODB_CONNECTION_STRING -ne '')"
} else {
    Write-Host "MongoDB connection string: not set (MongoDB features will be disabled)"
}

# Check if streamlit is available
try {
    $streamlitVersion = streamlit --version 2>&1
    Write-Host "Streamlit found: $streamlitVersion"
} catch {
    Write-Host "ERROR: Streamlit not found. Install with: pip install streamlit" -ForegroundColor Red
    exit 1
}

# Run Streamlit
Write-Host "Launching Streamlit on http://localhost:8501" -ForegroundColor Green
streamlit run app.py

