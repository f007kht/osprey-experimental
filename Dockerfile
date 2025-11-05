FROM python:3.11-slim-bookworm

# Install system dependencies AND detect TESSDATA_PREFIX
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-eng \
    ca-certificates \
    libssl-dev \
    openssl \
    && update-ca-certificates \
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
# Store it for use by entrypoint.sh
RUN TESSDATA_PREFIX_VALUE=$(cat /tmp/tessdata_prefix) && \
    echo "export TESSDATA_PREFIX=$TESSDATA_PREFIX_VALUE" >> /root/.bashrc && \
    echo "TESSDATA_PREFIX=$TESSDATA_PREFIX_VALUE" >> /etc/environment && \
    echo "$TESSDATA_PREFIX_VALUE" > /tmp/tessdata_prefix && \
    echo "TESSDATA_PREFIX (detected): $TESSDATA_PREFIX_VALUE"

# Set environment variables
ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV OMP_NUM_THREADS=4

# Copy Streamlit config
COPY .streamlit .streamlit

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Copy app files
COPY app.py .
COPY app_settings.py .
COPY helpers_mongo.py .

# Expose Streamlit port (use $PORT at runtime, default 8501)
EXPOSE 8501

# Run Streamlit with entrypoint that sets TESSDATA_PREFIX and uses $PORT
CMD ["/bin/bash", "/entrypoint.sh"]
