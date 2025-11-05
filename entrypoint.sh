#!/usr/bin/env bash
# Docker entrypoint script for Streamlit application
# Sets up TESSDATA_PREFIX and runs Streamlit with platform port support

set -euo pipefail

# Export TESSDATA_PREFIX from detected path or use default
if [ -f /tmp/tessdata_prefix ]; then
    export TESSDATA_PREFIX=$(cat /tmp/tessdata_prefix)
elif [ -f /etc/profile.d/tessdata.sh ]; then
    source /etc/profile.d/tessdata.sh 2>/dev/null || true
fi

# Use default if still not set
export TESSDATA_PREFIX="${TESSDATA_PREFIX:-/usr/share/tesseract-ocr/4.00/tessdata}"

# Run Streamlit with platform port support ($PORT for Spaces, Render, Fly, Heroku, etc.)
# Default to 8501 if PORT is not set
exec streamlit run app.py --server.address=0.0.0.0 --server.port="${PORT:-8501}"

