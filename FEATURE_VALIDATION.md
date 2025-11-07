# Feature Validation Checklist

This document provides a comprehensive checklist for validating all document processing features before scaling to a larger compute environment.

## Overview

This validation ensures that all Docling processing features work correctly in the current environment, so we can confidently scale to larger capacity compute with all features enabled.

## Current Environment

- **Platform:** Hugging Face Spaces
- **RAM:** 16 GB
- **CPU:** Multi-core (CPU-only, no GPU)
- **Compute Type:** Basic (free tier)

## Target Environment (Future Scaling)

- **Platform:** Hugging Face Spaces (upgraded tier) or dedicated infrastructure
- **RAM:** 32+ GB
- **CPU/GPU:** Higher core count or GPU acceleration
- **Compute Type:** Production-grade

---

## Feature Matrix

### 1. Image Extraction & Generation

| Feature | Environment Variable | UI Control | Default | Performance Impact | Status |
|---------|---------------------|------------|---------|-------------------|--------|
| **Page Image Generation** | `IMAGES_ENABLE=true` | N/A | ✅ Enabled | HIGH (3-5x slower) | ⏳ Testing |
| **Picture Image Generation** | `IMAGES_ENABLE=true` | N/A | ✅ Enabled | HIGH (per picture) | ⏳ Testing |
| **Image Scale** | `IMAGES_SCALE=0.75` | N/A | 0.75 (75%) | Proportional to scale² | ⏳ Testing |

**Validation Steps:**
- [ ] Process PDF with embedded images
- [ ] Verify page images are generated (check output)
- [ ] Verify picture images are extracted
- [ ] Test different scales (0.3, 0.5, 0.75, 1.0)
- [ ] Measure memory usage at each scale
- [ ] Measure processing time at each scale

---

### 2. Document Processing Features

| Feature | Environment Variable | UI Control | Default | Performance Impact | Status |
|---------|---------------------|------------|---------|-------------------|--------|
| **OCR (Text Extraction)** | N/A | Always on | ✅ Enabled | MEDIUM | ⏳ Testing |
| **Formula Enrichment (LaTeX)** | N/A | ☑️ Checkbox | ❌ Disabled | MEDIUM | ⏳ Testing |
| **Table Structure** | N/A | ☑️ Checkbox | ✅ Enabled | LOW-MEDIUM | ⏳ Testing |
| **Code Language Detection** | N/A | ☑️ Checkbox | ❌ Disabled | LOW | ⏳ Testing |
| **Picture Classification** | N/A | ☑️ Checkbox | ❌ Disabled | MEDIUM-HIGH | ⏳ Testing |

**Validation Steps:**

#### OCR (Always Enabled)
- [ ] Process scanned PDF (image-only, no text layer)
- [ ] Process native PDF (with text layer)
- [ ] Verify text extraction quality
- [ ] Test with text layer detection (should skip OCR when text layer exists)

#### Formula Enrichment
- [ ] Process PDF with mathematical formulas
- [ ] Enable "Extract LaTeX Formulas" checkbox in UI
- [ ] Verify LaTeX representations in output
- [ ] Check format: `$formula$` or `$$formula$$`

#### Table Structure
- [ ] Process PDF with tables
- [ ] Enable "Enhanced Table Structure" checkbox
- [ ] Verify table cells, rows, columns are extracted
- [ ] Check Markdown table formatting

#### Code Language Detection
- [ ] Process PDF with code blocks
- [ ] Enable "Code Language Detection" checkbox
- [ ] Verify programming language is detected
- [ ] Check code fence formatting: ```python, ```java, etc.

#### Picture Classification
- [ ] Process PDF with charts, diagrams, photos
- [ ] Enable "Picture Classification" checkbox
- [ ] Verify picture types are classified (chart, diagram, logo, photo, etc.)
- [ ] Check classification accuracy

---

### 3. Text Normalization Features

| Feature | Environment Variable | UI Control | Default | Performance Impact | Status |
|---------|---------------------|------------|---------|-------------------|--------|
| **Apply Text Normalization** | N/A | ☑️ Master checkbox | ❌ Disabled | LOW | ⏳ Testing |
| **Fix Spacing Issues** | N/A | ☑️ Sub-checkbox | ✅ Enabled (when normalization on) | MINIMAL | ⏳ Testing |
| **Replace Ligatures** | N/A | ☑️ Sub-checkbox | ✅ Enabled (when normalization on) | MINIMAL | ⏳ Testing |
| **Filter OCR Artifacts** | N/A | ☑️ Sub-checkbox | ❌ Disabled | MINIMAL | ⏳ Testing |

**Validation Steps:**

#### Text Normalization
- [ ] Process PDF with spacing issues (double spaces, irregular line breaks)
- [ ] Enable "Apply Text Normalization" checkbox
- [ ] Enable all sub-options (Fix Spacing, Replace Ligatures, Filter OCR Artifacts)
- [ ] Verify output has:
  - Single spaces (not double/triple)
  - Standard characters instead of ligatures (fi, fl instead of ﬁ, ﬂ)
  - Removed junk characters from OCR
  - Clean punctuation-only lines removed

---

### 4. Memory & Performance Tuning

| Parameter | Environment Variable | Default | Valid Range | Purpose | Status |
|-----------|---------------------|---------|-------------|---------|--------|
| **Pipeline Queue Max** | `PIPELINE_QUEUE_MAX=3` | 3 | 1-4 | Controls concurrent page processing | ⏳ Testing |
| **Image Scale** | `IMAGES_SCALE=0.75` | 0.75 | 0.0-1.0 | Controls image resolution | ⏳ Testing |
| **OCR Batch Size** | N/A (hardcoded) | 1 | 1-10 | Pages per OCR batch | ✅ Set to 1 |
| **Layout Batch Size** | N/A (hardcoded) | 1 | 1-10 | Pages per layout batch | ✅ Set to 1 |
| **Table Batch Size** | N/A (hardcoded) | 1 | 1-10 | Tables per batch | ✅ Set to 1 |

**Validation Steps:**
- [ ] Test with `PIPELINE_QUEUE_MAX=1` (slowest, lowest memory)
- [ ] Test with `PIPELINE_QUEUE_MAX=3` (current default)
- [ ] Test with `PIPELINE_QUEUE_MAX=4` (maximum)
- [ ] Monitor memory usage at each setting
- [ ] Measure processing time at each setting
- [ ] Find optimal queue size for current environment

---

### 5. MongoDB Storage & RAG Features

| Feature | Environment Variable | Default | Status |
|---------|---------------------|---------|--------|
| **MongoDB Storage** | `ENABLE_MONGODB=true` | ❌ Disabled | ⏳ Testing |
| **Embedding Model** | `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` | Local model | ⏳ Testing |
| **Remote Embeddings (VoyageAI)** | `USE_REMOTE_EMBEDDINGS=true` | ❌ Disabled | ⏳ Testing |

**Validation Steps:**
- [ ] Enable MongoDB in environment variables
- [ ] Process document and verify storage
- [ ] Verify vector embeddings are created
- [ ] Test semantic search functionality
- [ ] Test with local embeddings (sentence-transformers)
- [ ] Test with remote embeddings (VoyageAI)

---

## Test Documents

Create a test suite with the following document types:

### PDF Documents
1. **Native PDF with text layer** (no OCR needed)
   - Text content only
   - Expected: Fast processing, text extracted directly

2. **Scanned PDF** (OCR required)
   - Image-only, no text layer
   - Expected: Slower, OCR activated

3. **PDF with formulas**
   - Mathematical equations
   - Expected: LaTeX extraction when enabled

4. **PDF with tables**
   - Complex table structures
   - Expected: Table structure extraction

5. **PDF with code blocks**
   - Programming code snippets
   - Expected: Language detection when enabled

6. **PDF with images**
   - Charts, diagrams, photos
   - Expected: Picture classification when enabled

7. **Large PDF** (100+ pages)
   - Stress test for memory
   - Expected: Stable processing without OOM

8. **PDF with OCR artifacts**
   - Poor scan quality, ligatures
   - Expected: Text normalization improves output

### Office Documents
9. **DOCX** - Word document
10. **PPTX** - PowerPoint presentation
11. **XLSX** - Excel spreadsheet

---

## Performance Benchmarks

### Current Configuration (IMAGES_ENABLE=true)

| Document Type | Pages | Processing Time | Memory Peak | Notes |
|--------------|-------|-----------------|-------------|-------|
| Native PDF (text only) | 18 | ⏳ Testing | ⏳ Testing | Currently processing |
| Scanned PDF | - | - | - | - |
| PDF with formulas | - | - | - | - |
| PDF with tables | - | - | - | - |
| Large PDF (100 pages) | - | - | - | - |

### Optimized Configuration (IMAGES_ENABLE=false)

| Document Type | Pages | Processing Time | Memory Peak | Notes |
|--------------|-------|-----------------|-------------|-------|
| Native PDF (text only) | - | - | - | - |
| Scanned PDF | - | - | - | - |
| PDF with formulas | - | - | - | - |
| PDF with tables | - | - | - | - |
| Large PDF (100 pages) | - | - | - | - |

---

## Scaling Strategy

### Phase 1: Validate All Features (Current Environment)
**Goal:** Ensure every feature works correctly with test documents

1. ✅ Keep current configuration (`IMAGES_ENABLE=true`)
2. ⏳ Let current 18-page PDF finish processing
3. ⏳ Test each feature systematically with test documents
4. ⏳ Document performance baselines
5. ⏳ Identify any features that fail or produce poor results

### Phase 2: Optimize for Current Environment
**Goal:** Find best settings for 16 GB RAM environment

1. Disable image generation for speed: `IMAGES_ENABLE=false`
2. Test all features without image overhead
3. Measure performance improvements
4. Document optimal settings for production use

### Phase 3: Plan for Scaled Environment
**Goal:** Design configuration for upgraded compute

1. Calculate resource requirements for all features enabled:
   - Memory: 32+ GB recommended
   - CPU: 8+ cores or GPU acceleration
   - Storage: SSD for temp files

2. Design environment variable presets:
   - **Speed Mode:** Images off, minimal features
   - **Balanced Mode:** Images at 0.5 scale, core features
   - **Quality Mode:** Images at 1.0 scale, all features enabled

3. Create auto-scaling triggers based on document size

### Phase 4: Production Deployment
**Goal:** Deploy to scaled environment with monitoring

1. Upgrade Hugging Face Spaces tier or migrate to dedicated infrastructure
2. Enable all features with monitoring
3. Set up performance alerts
4. Create feature toggle UI for users

---

## Monitoring & Alerts

### Metrics to Track

1. **Processing Time**
   - Per page
   - Per document
   - By feature (OCR, image gen, etc.)

2. **Memory Usage**
   - Peak memory per document
   - Average memory per page
   - Memory by feature

3. **Quality Metrics**
   - Text extraction accuracy
   - Table structure completeness
   - Formula recognition rate
   - Picture classification accuracy

4. **Error Rates**
   - OSD failures
   - OCR failures
   - Timeout rate
   - OOM errors

### Current Monitoring (QA Dashboard)

The app already has monitoring via:
- QA Dashboard (Streamlit page)
- MongoDB quality metrics
- Logging with correlation IDs
- Alert system (`scripts/alerts_watch.py`)

---

## Validation Progress

**Last Updated:** 2025-11-07

### Overall Status: 🟡 IN PROGRESS

- ✅ Feature documentation complete
- ✅ Environment configuration documented
- ⏳ Test document suite creation
- ⏳ Feature validation testing
- ⏳ Performance benchmarking
- ⏳ Scaling strategy planning

### Next Steps

1. **Wait for current 18-page PDF to complete** ⏳
   - Monitor processing time
   - Check memory usage
   - Verify output quality
   - Check if image generation completed

2. **Create test document suite**
   - Upload or generate test PDFs for each feature
   - Organize in `tests/validation/` directory

3. **Systematic feature testing**
   - Test each feature individually
   - Test feature combinations
   - Document results in this file

4. **Performance benchmarking**
   - Measure baseline performance
   - Test with features disabled
   - Find optimal configuration

5. **Scaling recommendations**
   - Calculate resource requirements
   - Design environment presets
   - Plan migration strategy

---

## Questions for Scaling Decision

Before upgrading compute:

1. **Which features are most critical for your use case?**
   - Image extraction?
   - Formula recognition?
   - Picture classification?
   - Text normalization?

2. **What document types will you process most?**
   - Native PDFs (fast)
   - Scanned PDFs (slow, OCR-heavy)
   - Office documents
   - Large documents (100+ pages)

3. **What's your quality vs. speed preference?**
   - Maximum quality (all features, slow)
   - Balanced (core features, moderate speed)
   - Speed priority (minimal features, fast)

4. **What's your budget for compute?**
   - Free tier (current, 16 GB)
   - Small upgrade (32 GB, ~$20-30/month)
   - Large upgrade (64+ GB, GPU, ~$100+/month)

---

## Notes

- Image generation is the **#1 performance bottleneck** (3-5x slowdown)
- Text layer detection works well (skips OCR when text layer exists)
- Current queue size (3) is conservative for memory stability
- All ML features require CPU compute (no GPU needed, but would help)
- MongoDB storage adds minimal overhead (~100ms per document)

