#!/bin/bash
# Quick script to check the most recent processed document's images

echo "Checking most recent document for image data..."
echo "=============================================="
echo ""

# Set MongoDB connection if provided as argument
if [ -n "$1" ]; then
    export MONGODB_CONNECTION_STRING="$1"
fi

# Run the check script
python3 scripts/check_images.py --latest
