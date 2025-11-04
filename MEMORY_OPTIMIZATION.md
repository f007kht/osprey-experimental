# Memory Optimization Guide for Docling Image Generation

This document explains the memory optimization settings applied to keep image generation enabled while reducing memory pressure during document processing.

## Overview

Image generation in Docling can be memory-intensive, especially for large documents with high-resolution images. This guide shows how to optimize settings to handle 1000kb+ files while keeping image generation enabled.

## Key Optimizations Applied

### 1. Reduced Queue Sizes (Most Important)

**Default:** `queue_max_size = 100`
**Optimized:** `queue_max_size = 10`

Large queues hold many pages with images in memory simultaneously. Reducing from 100 to 10 prevents up to 100 pages with images from queuing up.

```python
pipeline_options.queue_max_size = 10  # Limit in-flight pages
```

### 2. Reduced Batch Sizes

**Default:** `batch_size = 4`
**Optimized:** `batch_size = 1`

Processing one page at a time reduces peak memory usage by avoiding accumulation of multiple pages with images.

```python
pipeline_options.ocr_batch_size = 1
pipeline_options.layout_batch_size = 1
pipeline_options.table_batch_size = 1
settings.perf.page_batch_size = 1
```

### 3. Reduced Image Scale

**Default:** `images_scale = 2.0`
**Optimized:** `images_scale = 1.5`

Reducing image scale from 2.0 to 1.5 reduces memory per image by approximately 44% (since memory scales with area: 1.5² vs 2.0²).

```python
pipeline_options.images_scale = 1.5  # Saves ~44% memory per image
```

### 4. Image Generation Enabled

Image generation is kept enabled for testing and verification:

```python
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True
```

## Expected Results

### Memory Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Queue Memory | ~100 pages × 3MB = 300MB | ~10 pages × 3MB = 30MB | -90% |
| Batch Memory | 4 pages × 3MB = 12MB | 1 page × 3MB = 3MB | -75% |
| Image Memory | 2.0x scale = 100% | 1.5x scale = 56% | -44% |
| **Total Peak** | ~400MB | ~120MB | **-70%** |

### Performance Impact

- **Processing Speed:** 10-20% slower (acceptable tradeoff for stability)
- **File Capacity:** Can handle 1000kb+ files reliably
- **Image Quality:** 1.5x scale provides good quality for most use cases

## Configuration Summary

All optimizations are applied in `app.py` in the `get_converter()` function:

```python
@st.cache_resource
def get_converter():
    pipeline_options = PdfPipelineOptions()
    
    # Enable image generation
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.images_scale = 1.5
    
    # Memory optimizations
    pipeline_options.queue_max_size = 10
    pipeline_options.ocr_batch_size = 1
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    
    # Global settings
    settings.perf.page_batch_size = 1
    
    return DocumentConverter(...)
```

## Monitoring Memory Usage

Use the `memory_monitor.py` utility to track memory consumption:

```bash
python memory_monitor.py your_document.pdf
```

This will show:
- Memory before conversion
- Memory after conversion
- Memory per page
- Memory when accessing images
- Peak memory usage

## Troubleshooting

### Still Running Out of Memory?

1. **Further reduce image scale:**
   ```python
   pipeline_options.images_scale = 1.0  # Minimum quality
   ```

2. **Process in page ranges:**
   ```python
   result = converter.convert("file.pdf", page_range=(1, 50))  # Process 50 pages at a time
   ```

3. **Disable image generation for very large files:**
   ```python
   pipeline_options.generate_page_images = False
   pipeline_options.generate_picture_images = False
   ```

### Performance Too Slow?

If memory is sufficient but processing is too slow, you can increase batch sizes:

```python
pipeline_options.queue_max_size = 20  # Increase from 10
pipeline_options.ocr_batch_size = 2   # Increase from 1
settings.perf.page_batch_size = 2     # Increase from 1
```

## Advanced: Processing Large Files in Chunks

For very large files (10MB+), consider processing in page ranges:

```python
def process_large_file(file_path, chunk_size=20):
    converter = get_converter()
    
    # Get total pages
    temp = converter.convert(file_path, page_range=(1, 1))
    total_pages = temp.input.page_count
    
    results = []
    for start in range(1, total_pages + 1, chunk_size):
        end = min(start + chunk_size - 1, total_pages)
        result = converter.convert(file_path, page_range=(start, end))
        results.append(result)
        # Process or save images here before next chunk
    
    return results
```

## References

- Docling Pipeline Options: `docling/datamodel/pipeline_options.py`
- Docling Settings: `docling/datamodel/settings.py`
- Memory Monitor: `memory_monitor.py`
- Troubleshooting Guide: `docs/usage/troubleshooting.md`

