# Heavy PDF Testing Plan

This document outlines the test plan for verifying that the application correctly processes and stores image+formula-heavy PDFs in MongoDB.

## Prerequisites

### Local Setup

```powershell
# Set environment variables
$env:ENABLE_MONGODB="true"
$env:MONGODB_CONNECTION_STRING="mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority"
$env:IMAGES_ENABLE="true"
$env:IMAGES_SCALE="0.75"
$env:PIPELINE_QUEUE_MAX="6"
$env:PAGE_BATCH_SIZE="1"

# Start the app
streamlit run app.py
```

### Hugging Face Spaces

Set the same environment variables in **Settings → Variables and secrets**:
- `ENABLE_MONGODB=true`
- `MONGODB_CONNECTION_STRING` (as a secret)
- `IMAGES_ENABLE=true`
- `IMAGES_SCALE=0.75`
- `PIPELINE_QUEUE_MAX=6`
- `PAGE_BATCH_SIZE=1`

The entrypoint already honors `$PORT` automatically.

## 1. Preflight Checks

### Health Check

```bash
# Local
curl "http://localhost:8501/?health=1"

# Expected JSON response:
{
  "status": "ok",
  "mongo": {
    "enabled": true,
    "status": "connected:ok:<hostname>"
  },
  "ocr": {
    "tessdata_prefix": "/usr/share/tesseract-ocr/4.00/tessdata"
  },
  "port": "8501",
  "version": "dev"
}
```

**Success criteria:**
- Response time < 100ms
- `status: "ok"`
- `mongo.enabled: true`
- `mongo.status` contains "connected" or "ok"

### MongoDB Connection

```bash
# Test MongoDB connection (if you have mongosh or mongo shell)
mongosh "<your-connection-string>"
# Should connect without errors
```

## 2. Test Document Battery

Upload three types of PDFs to test different scenarios:

### Test Case A: Text-Native, Figure-Heavy PDF

**Characteristics:**
- Vector-based formulas (not scanned)
- Multiple images/figures
- Selectable text (OCR mostly skipped)

**Expected Results:**
- ✅ Fast processing (OCR mostly skipped)
- ✅ Images rendered successfully
- ✅ Formulas extracted (check `docling_json` for `formula` nodes)
- ✅ MongoDB save succeeds
- ✅ No BSON size errors

**Verification:**
```bash
python test_mongodb_storage.py --analyze "test_case_a.pdf"
# Check: has_formulas: ✓, has_pictures: ✓, was_split: false (unless very large)
```

### Test Case B: Scanned, Formula-Heavy PDF

**Characteristics:**
- Raster/scanned pages
- Mathematical formulas in images
- Requires OCR

**Expected Results:**
- ⚠️ Slower processing (OCR required)
- ✅ Steady progress (no hangs)
- ✅ RAM stays comfortable with `IMAGES_SCALE=0.75` + queue 6
- ✅ MongoDB save succeeds
- ✅ Formulas extracted via OCR

**Verification:**
```bash
python test_mongodb_storage.py --analyze "test_case_b.pdf"
# Check: has_formulas: ✓, OCR processing evident in logs
```

### Test Case C: Short but Large File Size

**Characteristics:**
- ≤ 10 pages
- Large file size (high-DPI images, 600-1200 DPI)
- Embedded high-resolution photos

**Expected Results:**
- ✅ Processing completes
- ✅ If memory issues occur, lower `IMAGES_SCALE` to `0.5` and retry
- ✅ MongoDB save succeeds (may trigger split if document > 16MB)

**Troubleshooting:**
If memory issues occur:
```powershell
$env:IMAGES_SCALE="0.5"  # Lower from 0.75
# Restart app and retry
```

## 3. MongoDB Verification

### Create Indexes (One-Time)

```bash
# Using the test script
python test_mongodb_storage.py --create-indexes --database docling_documents --collection documents

# Or manually in MongoDB Shell/Compass:
db.documents.createIndex({ processed_at: -1 })
db.documents_pages.createIndex({ parent_id: 1 })
db.documents_pages.createIndex({ parent_id: 1, page_index: 1 }, { unique: true })
```

### Verify Document Storage

```bash
# List recent documents
python test_mongodb_storage.py --list --limit 10

# Analyze specific document
python test_mongodb_storage.py --analyze "YOUR_FILE.pdf"
```

### Manual MongoDB Queries

```javascript
// Primary document existence
db.documents.find(
  { original_filename: "YOUR_FILE.pdf" },
  { 
    original_filename: 1, 
    file_size: 1, 
    processed_at: 1, 
    "metadata.embedding_model": 1,
    "metadata.total_chunks": 1
  }
).limit(1)

// Check if document was split
const doc = db.documents.findOne({ original_filename: "YOUR_FILE.pdf" })
db.documents_pages.countDocuments({ parent_id: doc._id })

// Check for truncation
db.documents_pages.find({ parent_id: doc._id, _truncated: true }).limit(3)
```

## 4. Content Verification

### Analyze Document Structure

```python
# Run the analysis script
python test_mongodb_storage.py --analyze "YOUR_FILE.pdf"

# Or use Python interactively:
from test_mongodb_storage import get_mongo_client, analyze_document
import os

client = get_mongo_client()
db = client[os.environ.get("MONGODB_DATABASE", "docling_documents")]
results = analyze_document(db, "documents", "YOUR_FILE.pdf")
print(results)
```

**Expected Output:**
- `has_formulas: True` (if PDF contains formulas)
- `has_pictures: True` (if PDF contains images)
- `has_tables: True` (if PDF contains tables)
- `node_counts` shows top node types (formula, picture, table, etc.)

## 5. Performance Tuning

### If UI Freezes Mid-Run

**Problem:** Too many pages in flight

**Solution:**
```powershell
$env:PIPELINE_QUEUE_MAX="4"  # Reduce from 6
$env:PAGE_BATCH_SIZE="1"     # Keep at 1
# Restart app
```

### If High Memory During Rendering

**Problem:** Image rendering consumes too much RAM

**Solution:**
```powershell
$env:IMAGES_SCALE="0.5"  # Lower from 0.75
# Restart app and retry
```

### If MongoDB Save Hangs/Fails

**Problem:** Document exceeds 16MB BSON limit

**Expected Behavior:**
- Split helper logs: `"Document split: primary (X bytes) + Y pages"`
- Check `documents_pages` collection for split pages
- Verify `parent_id` links are correct

**If Still Failing:**
- Consider moving image arrays to GridFS (future enhancement)
- Check MongoDB Atlas connection limits
- Verify network connectivity

### If OCR Takes Too Long on Text-Native PDFs

**Future Enhancement:**
- Add "text present?" gate to skip OCR on text-native PDFs
- Peek first pages with `pypdf`
- Disable OCR if selectable text exists

## 6. Success Criteria

### Acceptance Gate

- ✅ Health check returns `status: ok` in < 100ms
- ✅ Each test PDF completes processing
- ✅ Images/formulas present in `docling_json`
- ✅ Data correctly split across `documents` + `documents_pages` (if needed)
- ✅ No BSON size errors
- ✅ `truncated: true` is rare (edge pages only)
- ✅ RAM doesn't creep unbounded
- ✅ CPU is busy but stable

### Metrics to Monitor

- **Health check response time**: Should be < 100ms
- **Processing time**: Varies by document size and OCR requirements
- **Memory usage**: Should stay within available RAM (16GB on HF Spaces)
- **MongoDB write time**: Should complete within seconds
- **Split frequency**: Should only occur for very large documents (>16MB)

## 7. Troubleshooting

### Collect Debug Information

If issues occur, collect:

1. **Health check JSON:**
   ```bash
   curl "http://localhost:8501/?health=1"
   ```

2. **Last 30 log lines around split:**
   ```bash
   # Check app logs for "split_for_mongo" messages
   # Look for: "Document split: primary (X bytes) + Y pages"
   ```

3. **MongoDB page counts:**
   ```bash
   python test_mongodb_storage.py --analyze "PROBLEM_FILE.pdf"
   ```

4. **Memory/CPU metrics:**
   - Check system resources during processing
   - Monitor for OOM (Out of Memory) errors

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Health check slow | Response > 100ms | Check early health check placement (should be before heavy imports) |
| UI freezes | No progress indicator | Lower `PIPELINE_QUEUE_MAX` to 4 |
| High memory | OOM errors | Lower `IMAGES_SCALE` to 0.5 |
| MongoDB save fails | BSON size errors | Verify split helper is working; check `documents_pages` collection |
| OCR too slow | Processing takes forever | Add skip-OCR heuristic (future enhancement) |

## 8. Future Enhancements

### Nice-to-Have (Next Iteration)

- **GridFS for image binaries**: Store page/picture crops; keep refs in pages
- **Async/off-thread processing**: Keep Streamlit extra snappy during long OCR
- **Skip-OCR heuristic**: Peek first pages with `pypdf`, disable OCR if selectable text exists
- **Metrics in logs**: Per-page time, OCR time, render time, Mongo insert durations

## 9. Test Script Usage

```bash
# Create indexes (one-time)
python test_mongodb_storage.py --create-indexes

# List recent documents
python test_mongodb_storage.py --list --limit 10

# Analyze specific document
python test_mongodb_storage.py --analyze "document.pdf"

# Full help
python test_mongodb_storage.py --help
```

## 10. Reporting Results

After testing, document:

1. **Test results** for each test case (A, B, C)
2. **Any issues encountered** and resolutions
3. **Performance metrics** (processing time, memory usage)
4. **MongoDB storage verification** (splits, truncations, content detection)
5. **Recommended tuning** (if any adjustments needed)

