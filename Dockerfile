FROM python:3.11-slim-bookworm

# Install system dependencies AND detect TESSDATA_PREFIX
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-eng \
    ca-certificates \
    libssl-dev \
    openssl \
    && rm -rf /var/lib/apt/lists/* \
    && TESSDATA_PATH=$(dpkg -L tesseract-ocr-eng | grep 'tessdata$' | head -n 1) \
    && if [ -n "$TESSDATA_PATH" ]; then \
        case "$TESSDATA_PATH" in \
          */) ;; \
          *) TESSDATA_PATH="$TESSDATA_PATH/" ;; \
        esac \
       fi \
    && echo "$TESSDATA_PATH" > /tmp/tessdata_prefix \
    && echo "export TESSDATA_PREFIX=$TESSDATA_PATH" >> /etc/profile.d/tessdata.sh \
    && chmod +x /etc/profile.d/tessdata.sh \
    && echo "TESSDATA_PREFIX (detected): $TESSDATA_PATH"

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
    -r requirements.txt

# Set TESSDATA_PREFIX environment variable from the detected path
# The path was detected during Tesseract installation and saved to /tmp/tessdata_prefix
# Create entrypoint script that exports TESSDATA_PREFIX before running Streamlit
# (ENV doesn't support command substitution, so we use a runtime wrapper)
RUN TESSDATA_PREFIX_VALUE=$(cat /tmp/tessdata_prefix) && \
    echo "export TESSDATA_PREFIX=$TESSDATA_PREFIX_VALUE" >> /root/.bashrc && \
    echo "TESSDATA_PREFIX=$TESSDATA_PREFIX_VALUE" >> /etc/environment && \
    printf '#!/bin/bash\nsource /etc/profile.d/tessdata.sh 2>/dev/null || true\nif [ -f /tmp/tessdata_prefix ]; then\n  export TESSDATA_PREFIX=$(cat /tmp/tessdata_prefix)\nfi\nexec streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Set environment variables
ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV OMP_NUM_THREADS=4

# Copy Streamlit config
COPY .streamlit .streamlit

# Copy app files
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit with entrypoint that sets TESSDATA_PREFIX
CMD ["/bin/bash", "/entrypoint.sh"]
