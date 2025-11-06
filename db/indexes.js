// MongoDB indexes for observability and quality gates
// Run with: mongosh <connection_string> < db/indexes.js
// Or: mongo <connection_string> < db/indexes.js

db = db.getSiblingDB('docling_db');  // Adjust database name as needed
collection = db.documents;

// Create indexes for quality observability
collection.createIndex({"input.format": 1});
collection.createIndex({"status.quality_bucket": 1});
collection.createIndex({"warnings.osd_fail_count": 1});
collection.createIndex({"metrics.process_seconds": -1});
collection.createIndex({"metrics.page_count": -1});
collection.createIndex({"metrics.markdown_length": -1});

// Compound index for common QA dashboard filters
collection.createIndex({"input.format": 1, "status.quality_bucket": 1, "metrics.process_seconds": -1});

// Index for idempotent upsert by content_hash
collection.createIndex({"content_hash": 1, "filename": 1}, {unique: false});

print("✓ Created indexes for quality observability");

