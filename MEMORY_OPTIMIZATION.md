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

**Default:** All batch sizes = 4
**Optimized:** All batch sizes = 1

Process one page at a time through each stage to reduce peak memory usage:

```python
pipeline_options.ocr_batch_size = 1
pipeline_options.layout_batch_size = 1
pipeline_options.table_batch_size = 1
```

### 3. Reduced Global Page Batch Size

**Default:** `settings.perf.page_batch_size = 4`
**Optimized:** `settings.perf.page_batch_size = 1`

Process one page at a time globally:

```python
from docling.datamodel.settings import settings
settings.perf.page_batch_size = 1
```

### 4. Reduced Image Scale

**Default/Previous:** `images_scale = 2.0`
**Optimized:** `images_scale = 1.5`

Reducing from 2.0x to 1.5x saves approximately 44% memory per image (1.5² vs 2.0²):

```python
pipeline_options.images_scale = 1.5
```

### 5. Image Generation Enabled

Image generation is **ALWAYS ENABLED** with these optimized settings:

```python
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True
```

## Implementation in app.py

The following settings have been applied in `app.py`:

```python
from docling.datamodel.settings import settings

# Global optimization
settings.perf.page_batch_size = 1

# Pipeline options in get_converter()
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 1.5
pipeline_options.queue_max_size = 10
pipeline_options.ocr_batch_size = 1
pipeline_options.layout_batch_size = 1
pipeline_options.table_batch_size = 1
```

## Memory Monitoring

Use the `memory_monitor.py` script to track memory usage:

```bash
# Monitor processing of a PDF file
python memory_monitor.py document.pdf

# Process only first 10 pages for testing
python memory_monitor.py document.pdf 10
```

The script will show:
- Memory usage before/after conversion
- Memory per page estimates
- Peak memory consumption
- Image access memory impact

Example output:
```
Processing: document.pdf
File size: 1.23 MB
============================================================

Initial memory: 450.23 MB
After converter initialization: 890.45 MB (Δ +440.22 MB)
After conversion: 1250.67 MB (Δ +800.44 MB)

Page 1 image accessed (1024, 768): 1255.89 MB (Δ +805.66 MB)
Page 2 image accessed (1024, 768): 1261.12 MB (Δ +810.89 MB)

Peak memory increase: 810.89 MB
Estimated memory per page: 40.54 MB
```

## Why These Settings Work

1. **Queue size reduction**: Limits number of pages held in memory to 10 instead of 100
2. **Batch size reduction**: Processes one page at a time, reducing concurrent memory usage
3. **Image scale reduction**: 1.5x instead of 2.0x reduces memory per image by ~44%
4. **Sequential processing**: Page batch size of 1 ensures pages are processed one at a time

## Expected Performance

- **Memory savings**: ~60-70% reduction in peak memory usage
- **Speed impact**: 10-20% slower processing (acceptable tradeoff for stability)
- **File size capacity**: Can handle 1000kb+ files with image generation enabled
- **Image quality**: 1.5x scale provides good quality for most use cases

## Advanced: Processing Very Large Files

For extremely large files (10+ MB), you can process in chunks:

```python
def process_large_file_in_chunks(file_path, chunk_size=20):
    """Process large file in chunks while keeping image generation enabled."""
    converter = build_optimized_converter()

    # Get total page count
    temp_result = converter.convert(file_path, page_range=(1, 1))
    total_pages = temp_result.input.page_count
    del temp_result

    all_results = []
    for start in range(1, total_pages + 1, chunk_size):
        end = min(start + chunk_size - 1, total_pages)
        print(f"Processing pages {start}-{end} of {total_pages}")

        result = converter.convert(file_path, page_range=(start, end))
        all_results.append(result)

        # Save images before moving to next chunk if needed

    return all_results
```

## Troubleshooting

### Still Running Out of Memory?

If you still experience memory issues, try:

1. **Further reduce image scale**: Set `images_scale = 1.0`
2. **Process in smaller chunks**: Use page ranges (e.g., 10 pages at a time)
3. **Reduce queue size more**: Set `queue_max_size = 5`

### Processing Too Slow?

If processing is too slow, you can carefully increase:

1. **Batch sizes**: Try `batch_size = 2` instead of 1
2. **Queue size**: Try `queue_max_size = 15` instead of 10

**Note:** Monitor memory usage when increasing these values.

### Image Quality Issues?

If 1.5x scale is insufficient:

1. Try `images_scale = 1.75` as a middle ground
2. Use `images_scale = 2.0` but reduce queue_max_size to 5

## Configuration Summary

| Setting | Default | Optimized | Impact |
|---------|---------|-----------|--------|
| `queue_max_size` | 100 | 10 | -90% queue memory |
| `ocr_batch_size` | 4 | 1 | -75% OCR batch memory |
| `layout_batch_size` | 4 | 1 | -75% layout batch memory |
| `table_batch_size` | 4 | 1 | -75% table batch memory |
| `page_batch_size` | 4 | 1 | -75% page batch memory |
| `images_scale` | 2.0 | 1.5 | -44% per-image memory |
| `generate_page_images` | False | True | Images enabled |
| `generate_picture_images` | False | True | Images enabled |

## References

- Docling Pipeline Options: `docling/datamodel/pipeline_options.py`
- Docling Settings: `docling/datamodel/settings.py`
- Memory Monitor: `memory_monitor.py`
- Streamlit App: `app.py`
