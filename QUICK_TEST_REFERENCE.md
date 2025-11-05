# Quick Test Reference Card

## Environment Variables (PowerShell)

```powershell
$env:ENABLE_MONGODB="true"
$env:MONGODB_CONNECTION_STRING="mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority"
$env:IMAGES_ENABLE="true"
$env:IMAGES_SCALE="0.75"
$env:PIPELINE_QUEUE_MAX="6"
$env:PAGE_BATCH_SIZE="1"
streamlit run app.py
```

## Health Check

```bash
curl "http://localhost:8501/?health=1"
# Expect: {"status":"ok","mongo":{"enabled":true,"status":"connected:ok:<host>"}}
```

## MongoDB Indexes (One-Time)

```bash
python test_mongodb_storage.py --create-indexes
```

## Verify Document Storage

```bash
# List recent
python test_mongodb_storage.py --list

# Analyze specific file
python test_mongodb_storage.py --analyze "YOUR_FILE.pdf"
```

## Tuning Knobs

| Issue | Solution |
|-------|----------|
| UI freezes | `$env:PIPELINE_QUEUE_MAX="4"` |
| High memory | `$env:IMAGES_SCALE="0.5"` |
| Save fails | Check split logs, verify `documents_pages` collection |

## Success Criteria

- ✅ Health check < 100ms
- ✅ PDFs complete processing
- ✅ Formulas/pictures in `docling_json`
- ✅ No BSON size errors
- ✅ RAM stable

## MongoDB Queries

```javascript
// Find document
db.documents.findOne({ original_filename: "FILE.pdf" })

// Count pages
db.documents_pages.countDocuments({ parent_id: <doc_id> })

// Check truncation
db.documents_pages.find({ parent_id: <doc_id>, _truncated: true })
```

