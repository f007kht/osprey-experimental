FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --find-links https://download.pytorch.org/whl/cpu/torch_stable.html \
    -r requirements.txt

# Set environment variables
ENV OPENCV_IO_ENABLE_OPENEXR=0
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata
ENV OMP_NUM_THREADS=4

# Copy Streamlit config
COPY .streamlit .streamlit

# Copy app files
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
