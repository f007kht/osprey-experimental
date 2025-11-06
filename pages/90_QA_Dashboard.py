"""QA Dashboard page for monitoring document conversion quality metrics."""

import os
import sys
import json
from pathlib import Path

import streamlit as st

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from app_settings import get_settings
    app_config = get_settings()
except ImportError:
    st.error("Failed to import app_settings")
    st.stop()

# Try to import MongoDB
try:
    from pymongo import MongoClient
    import certifi
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False


def get_mongo_client(connection_string: str):
    """Create MongoDB client with SSL/TLS configuration."""
    if not PYMONGO_AVAILABLE:
        return None
    return MongoClient(
        connection_string,
        tls=True,
        tlsCAFile=certifi.where(),
        tlsAllowInvalidCertificates=False,
        tlsAllowInvalidHostnames=False,
        serverSelectionTimeoutMS=10000,
        connectTimeoutMS=10000,
        socketTimeoutMS=10000,
    )


st.set_page_config(page_title="QA Dashboard", layout="wide")

st.title("📊 QA Dashboard")
st.write("Monitor document conversion quality metrics and statistics")

# Show active feature flags and guardrail limits
st.subheader("⚙️ Configuration")
col_flag1, col_flag2, col_flag3, col_flag4 = st.columns(4)

import os
qa_flags = {
    "PDF Warning Suppress": os.getenv("QA_FLAG_ENABLE_PDF_WARNING_SUPPRESS", "1") == "1",
    "Text Layer Detect": os.getenv("QA_FLAG_ENABLE_TEXT_LAYER_DETECT", "1") == "1",
    "Log Normalized Codes": os.getenv("QA_FLAG_LOG_NORMALIZED_CODES", "1") == "1",
}
qa_limits = {
    "Max Pages": int(os.getenv("QA_MAX_PAGES", "500")),
    "Max Seconds": float(os.getenv("QA_MAX_SECONDS", "300")),
    "Schema Version": int(os.getenv("QA_SCHEMA_VERSION", "2")),
}

with col_flag1:
    st.caption("Feature Flags")
    for flag, enabled in qa_flags.items():
        st.write(f"{'✅' if enabled else '❌'} {flag}")

with col_flag2:
    st.caption("Guardrail Limits")
    for limit, value in qa_limits.items():
        st.write(f"🔒 {limit}: {value}")

with col_flag3:
    st.caption("MongoDB")
    st.write(f"📊 Database: {database_name}")
    st.write(f"📁 Collection: {collection_name}")

with col_flag4:
    st.caption("Quick Actions")
    if st.button("🔄 Refresh"):
        st.rerun()

# Check MongoDB availability
if not PYMONGO_AVAILABLE:
    st.warning("⚠️ pymongo not available. Install with: `pip install pymongo[srv]`")
    st.stop()

if not app_config.enable_mongodb:
    st.warning("⚠️ MongoDB is disabled. Enable it in app configuration.")
    st.stop()

connection_string = app_config.mongodb_connection_string
database_name = app_config.mongodb_database
collection_name = app_config.mongodb_collection

if not connection_string:
    st.warning("⚠️ MongoDB connection string not configured.")
    st.info("Set MONGODB_CONNECTION_STRING environment variable")
    st.stop()

# Connect to MongoDB (fail soft)
try:
    client = get_mongo_client(connection_string)
    db = client[database_name]
    collection = db[collection_name]
    
    # Test connection
    client.admin.command('ping')
    st.success("✓ Connected to MongoDB")
except Exception as e:
    st.error(f"❌ Failed to connect to MongoDB: {e}")
    st.info("The dashboard will not function without a valid MongoDB connection.")
    st.stop()

# Top counters
st.header("📈 Overview")

col1, col2, col3, col4 = st.columns(4)

try:
    total_docs = collection.count_documents({})
    col1.metric("Total Documents", total_docs)
    
    # Count by format
    format_counts = {}
    for doc in collection.find({}, {"input.format": 1}):
        fmt = doc.get("input", {}).get("format", "unknown")
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    col2.metric("Formats", len(format_counts))
    
    # Bucket counts
    ok_count = collection.count_documents({"status.quality_bucket": "ok"})
    suspect_count = collection.count_documents({"status.quality_bucket": "suspect"})
    fail_count = collection.count_documents({"status.quality_bucket": "fail"})
    
    col3.metric("OK", ok_count)
    col4.metric("Suspect", suspect_count)
    
except Exception as e:
    st.error(f"Error fetching overview: {e}")

# Format breakdown
st.header("📋 Documents by Format")

try:
    format_data = {}
    for doc in collection.find({}, {"input.format": 1}):
        fmt = doc.get("input", {}).get("format", "unknown")
        format_data[fmt] = format_data.get(fmt, 0) + 1
    
    if format_data:
        st.bar_chart(format_data)
    else:
        st.info("No documents found")
except Exception as e:
    st.error(f"Error fetching format data: {e}")

# Processing time stats
st.header("⏱️ Processing Time Statistics")

try:
    # Get process_seconds by format
    pipeline = [
        {
            "$match": {
                "metrics.process_seconds": {"$ne": None, "$exists": True}
            }
        },
        {
            "$group": {
                "_id": "$input.format",
                "count": {"$sum": 1},
                "avg": {"$avg": "$metrics.process_seconds"},
                "min": {"$min": "$metrics.process_seconds"},
                "max": {"$max": "$metrics.process_seconds"},
            }
        },
        {
            "$sort": {"count": -1}
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    
    if results:
        import pandas as pd
        df = pd.DataFrame([
            {
                "Format": r["_id"],
                "Count": r["count"],
                "Avg (s)": round(r["avg"], 2),
                "Min (s)": round(r["min"], 2),
                "Max (s)": round(r["max"], 2),
            }
            for r in results
        ])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No processing time data available")
except Exception as e:
    st.error(f"Error fetching processing time stats: {e}")

# Suspect documents table
st.header("⚠️ Suspect Documents")

# Filters
col_filter1, col_filter2 = st.columns(2)

with col_filter1:
    format_filter = st.selectbox(
        "Filter by Format",
        options=["All"] + list(format_counts.keys()) if format_counts else ["All"],
        index=0
    )

with col_filter2:
    bucket_filter = st.selectbox(
        "Filter by Bucket",
        options=["All", "suspect", "fail"],
        index=0
    )

try:
    # Build query
    query = {}
    if format_filter != "All":
        query["input.format"] = format_filter
    if bucket_filter != "All":
        query["status.quality_bucket"] = bucket_filter
    else:
        # Show suspect and fail if "All" selected
        query["status.quality_bucket"] = {"$in": ["suspect", "fail"]}
    
    suspect_docs = list(collection.find(
        query,
        {
            "filename": 1,
            "input.format": 1,
            "metrics.markdown_length": 1,
            "metrics.page_count": 1,
            "warnings.osd_fail_count": 1,
            "warnings.wmf_missing_loader": 1,
            "warnings.format_conflict": 1,
            "text_layer_detected": 1,
            "processed_at": 1,
            "run_id": 1,
            "content_hash": 1,
            "status.abort": 1,
        }
    ).sort("processed_at", -1).limit(50))
    
    if suspect_docs:
        import pandas as pd
        df = pd.DataFrame([
            {
                "Filename": doc.get("filename", "unknown"),
                "Format": doc.get("input", {}).get("format", "unknown"),
                "MD Length": doc.get("metrics", {}).get("markdown_length", 0),
                "Pages": doc.get("metrics", {}).get("page_count", "?"),
                "OSD Fails": doc.get("warnings", {}).get("osd_fail_count", 0),
                "WMF Missing": "Yes" if doc.get("warnings", {}).get("wmf_missing_loader") else "No",
                "Format Conflict": "Yes" if doc.get("warnings", {}).get("format_conflict") else "No",
                "Text Layer": "Yes" if doc.get("text_layer_detected") else "No",
                "Run ID": doc.get("run_id", "")[:8] if doc.get("run_id") else "",
                "Content Hash": doc.get("content_hash", "")[:8] if doc.get("content_hash") else "",
                "Abort Reason": doc.get("status", {}).get("abort", {}).get("reason", "") if doc.get("status", {}).get("abort") else "",
            }
            for doc in suspect_docs
        ])
        st.dataframe(df, use_container_width=True)
        
        # Copy query button
        st.caption("💡 Tip: Use run_id or content_hash to join with QA logs")
        query_json = json.dumps(query, indent=2)
        st.code(query_json, language="json")
        st.button("📋 Copy Query", key="copy_query", help="Copy the MongoDB query to clipboard")
    else:
        st.info("No suspect documents found")
except Exception as e:
    st.error(f"Error fetching suspect documents: {e}")

# Close connection
try:
    client.close()
except Exception:
    pass

