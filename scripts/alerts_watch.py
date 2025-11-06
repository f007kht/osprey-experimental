#!/usr/bin/env python3
"""
Alerting system that monitors MongoDB quality metrics and sends notifications.

Runs threshold checks every N seconds and sends alerts via webhook or email
when quality metrics breach configured thresholds.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

try:
    from pymongo import MongoClient
    import certifi
    PYMONGO_AVAILABLE = True
except ImportError:
    print("ERROR: pymongo not installed. Install with: pip install pymongo[srv]")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration from environment
ALERT_INTERVAL_SECONDS = int(os.getenv("ALERT_INTERVAL_SECONDS", "300"))
ALERT_SUSPECT_RATE = float(os.getenv("ALERT_SUSPECT_RATE", "0.2"))
ALERT_OSD_SPIKE_MULT = float(os.getenv("ALERT_OSD_SPIKE_MULT", "3.0"))
ALERT_MD_P10_DROP = float(os.getenv("ALERT_MD_P10_DROP", "100"))
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO")
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "alerts@docling.local")
ALERT_SMTP_HOST = os.getenv("ALERT_SMTP_HOST")
ALERT_SMTP_PORT = int(os.getenv("ALERT_SMTP_PORT", "587"))
ALERT_SMTP_USER = os.getenv("ALERT_SMTP_USER")
ALERT_SMTP_PASS = os.getenv("ALERT_SMTP_PASS")

MONGODB_CONNECTION_STRING = os.getenv("MONGODB_CONNECTION_STRING")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "docling_db")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "documents")


def get_mongo_client(connection_string: str):
    """Create MongoDB client with SSL/TLS configuration."""
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


def send_webhook_alert(webhook_url: str, payload: Dict[str, Any]) -> bool:
    """Send alert via webhook (POST JSON)."""
    try:
        import requests
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Webhook alert failed: {e}")
        return False


def send_email_alert(to_email: str, subject: str, body: str) -> bool:
    """Send alert via email (SMTP)."""
    if not all([ALERT_SMTP_HOST, ALERT_SMTP_USER, ALERT_SMTP_PASS]):
        logging.warning("Email alert configured but SMTP settings incomplete")
        return False
    
    server = None
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart()
        msg['From'] = ALERT_EMAIL_FROM
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(ALERT_SMTP_HOST, ALERT_SMTP_PORT)
        server.starttls()
        server.login(ALERT_SMTP_USER, ALERT_SMTP_PASS)
        server.send_message(msg)
        return True
    except Exception as e:
        logging.error(f"Email alert failed: {e}")
        return False
    finally:
        # Ensure SMTP connection is always closed, even on exception
        if server is not None:
            try:
                server.quit()
            except Exception:
                pass  # Ignore errors during cleanup


def check_suspect_rate(collection, threshold: float) -> Optional[Dict[str, Any]]:
    """Check if suspect rate by format exceeds threshold (last 24h)."""
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    
    pipeline = [
        {
            "$match": {
                "processed_at": {"$gte": last_24h.isoformat()}
            }
        },
        {
            "$group": {
                "_id": "$input.format",
                "total": {"$sum": 1},
                "suspect": {
                    "$sum": {"$cond": [{"$eq": ["$status.quality_bucket", "suspect"]}, 1, 0]}
                }
            }
        },
        {
            "$project": {
                "format": "$_id",
                "total": 1,
                "suspect": 1,
                "rate": {
                    "$cond": [
                        {"$eq": ["$total", 0]},
                        0,
                        {"$divide": ["$suspect", "$total"]}
                    ]
                }
            }
        },
        {
            "$match": {"rate": {"$gt": threshold}}
        }
    ]
    
    results = list(collection.aggregate(pipeline))
    if results:
        return {
            "check": "suspect_rate",
            "threshold": threshold,
            "breaches": [
                {
                    "format": r["format"],
                    "rate": r["rate"],
                    "suspect": r["suspect"],
                    "total": r["total"]
                }
                for r in results
            ]
        }
    return None


def check_osd_spike(collection, multiplier: float) -> Optional[Dict[str, Any]]:
    """Check if OSD fail count spikes (1h mean > 24h mean × multiplier)."""
    now = datetime.utcnow()
    last_1h = now - timedelta(hours=1)
    last_24h = now - timedelta(hours=24)
    
    # 24h mean
    pipeline_24h = [
        {
            "$match": {
                "processed_at": {"$gte": last_24h.isoformat()},
                "warnings.osd_fail_count": {"$exists": True}
            }
        },
        {
            "$group": {
                "_id": None,
                "mean": {"$avg": "$warnings.osd_fail_count"}
            }
        }
    ]
    result_24h = list(collection.aggregate(pipeline_24h))
    mean_24h = result_24h[0]["mean"] if result_24h and result_24h[0].get("mean") else 0
    
    # 1h mean
    pipeline_1h = [
        {
            "$match": {
                "processed_at": {"$gte": last_1h.isoformat()},
                "warnings.osd_fail_count": {"$exists": True}
            }
        },
        {
            "$group": {
                "_id": None,
                "mean": {"$avg": "$warnings.osd_fail_count"}
            }
        }
    ]
    result_1h = list(collection.aggregate(pipeline_1h))
    mean_1h = result_1h[0]["mean"] if result_1h and result_1h[0].get("mean") else 0
    
    if mean_24h > 0 and mean_1h > mean_24h * multiplier:
        return {
            "check": "osd_spike",
            "threshold_multiplier": multiplier,
            "mean_24h": mean_24h,
            "mean_1h": mean_1h,
            "spike_ratio": mean_1h / mean_24h if mean_24h > 0 else 0
        }
    return None


def check_markdown_density_drop(collection, drop_threshold: float) -> Optional[Dict[str, Any]]:
    """Check if markdown density p10 drops significantly (1h < 24h - threshold)."""
    now = datetime.utcnow()
    last_1h = now - timedelta(hours=1)
    last_24h = now - timedelta(hours=24)
    
    # 24h p10
    pipeline_24h = [
        {
            "$match": {
                "processed_at": {"$gte": last_24h.isoformat()},
                "metrics.page_count": {"$ne": None, "$gt": 0},
                "metrics.markdown_length": {"$ne": None, "$exists": True}
            }
        },
        {
            "$addFields": {
                "md_per_page": {
                    "$divide": ["$metrics.markdown_length", "$metrics.page_count"]
                }
            }
        },
        {
            "$group": {
                "_id": None,
                "p10": {"$percentile": {"input": "$md_per_page", "p": [0.1], "method": "approximate"}}
            }
        }
    ]
    result_24h = list(collection.aggregate(pipeline_24h))
    p10_24h = result_24h[0]["p10"][0] if result_24h and result_24h[0].get("p10") else 0
    
    # 1h p10
    pipeline_1h = [
        {
            "$match": {
                "processed_at": {"$gte": last_1h.isoformat()},
                "metrics.page_count": {"$ne": None, "$gt": 0},
                "metrics.markdown_length": {"$ne": None, "$exists": True}
            }
        },
        {
            "$addFields": {
                "md_per_page": {
                    "$divide": ["$metrics.markdown_length", "$metrics.page_count"]
                }
            }
        },
        {
            "$group": {
                "_id": None,
                "p10": {"$percentile": {"input": "$md_per_page", "p": [0.1], "method": "approximate"}}
            }
        }
    ]
    result_1h = list(collection.aggregate(pipeline_1h))
    p10_1h = result_1h[0]["p10"][0] if result_1h and result_1h[0].get("p10") else 0
    
    if p10_24h > 0 and p10_1h < (p10_24h - drop_threshold):
        return {
            "check": "markdown_density_drop",
            "drop_threshold": drop_threshold,
            "p10_24h": p10_24h,
            "p10_1h": p10_1h,
            "drop": p10_24h - p10_1h
        }
    return None


def get_top_offenders(collection, limit: int = 5) -> List[Dict[str, Any]]:
    """Get top N suspect documents with run_id and filename."""
    now = datetime.utcnow()
    last_24h = now - timedelta(hours=24)
    
    docs = list(collection.find(
        {
            "processed_at": {"$gte": last_24h.isoformat()},
            "status.quality_bucket": {"$in": ["suspect", "fail"]}
        },
        {
            "run_id": 1,
            "filename": 1,
            "input.format": 1,
            "status.quality_bucket": 1,
            "status.abort": 1
        }
    ).sort("processed_at", -1).limit(limit))
    
    return [
        {
            "run_id": doc.get("run_id", "")[:8] if doc.get("run_id") else "unknown",
            "filename": doc.get("filename", "unknown"),
            "format": doc.get("input", {}).get("format", "unknown"),
            "bucket": doc.get("status", {}).get("quality_bucket", "unknown"),
            "abort": doc.get("status", {}).get("abort", {}).get("reason", "") if doc.get("status", {}).get("abort") else ""
        }
        for doc in docs
    ]


def run_checks(collection) -> List[Dict[str, Any]]:
    """Run all threshold checks and return breaches."""
    breaches = []
    
    # Check suspect rate
    suspect_breach = check_suspect_rate(collection, ALERT_SUSPECT_RATE)
    if suspect_breach:
        breaches.append(suspect_breach)
    
    # Check OSD spike
    osd_breach = check_osd_spike(collection, ALERT_OSD_SPIKE_MULT)
    if osd_breach:
        breaches.append(osd_breach)
    
    # Check markdown density drop
    md_breach = check_markdown_density_drop(collection, ALERT_MD_P10_DROP)
    if md_breach:
        breaches.append(md_breach)
    
    return breaches


def main():
    """Main alerting loop."""
    if not MONGODB_CONNECTION_STRING:
        logging.error("MONGODB_CONNECTION_STRING not set")
        sys.exit(1)
    
    if not ALERT_WEBHOOK_URL and not ALERT_EMAIL_TO:
        logging.warning("No alert destinations configured (ALERT_WEBHOOK_URL or ALERT_EMAIL_TO)")
        logging.info("Continuing in monitoring mode (no alerts will be sent)")
    
    logging.info(f"Starting alerts watch (interval: {ALERT_INTERVAL_SECONDS}s)")
    logging.info(f"Thresholds: suspect_rate={ALERT_SUSPECT_RATE}, osd_spike={ALERT_OSD_SPIKE_MULT}x, md_drop={ALERT_MD_P10_DROP}")
    
    try:
        client = get_mongo_client(MONGODB_CONNECTION_STRING)
        db = client[MONGODB_DATABASE]
        collection = db[MONGODB_COLLECTION]
        
        # Test connection
        client.admin.command('ping')
        logging.info("✓ Connected to MongoDB")
        
        last_alert_time = {}  # Track last alert time per check type
        
        while True:
            try:
                breaches = run_checks(collection)
                
                if breaches:
                    # Get top offenders
                    top_offenders = get_top_offenders(collection, limit=5)
                    
                    # Build alert payload
                    alert_payload = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "breaches": breaches,
                        "top_offenders": top_offenders,
                        "config": {
                            "suspect_rate_threshold": ALERT_SUSPECT_RATE,
                            "osd_spike_multiplier": ALERT_OSD_SPIKE_MULT,
                            "md_drop_threshold": ALERT_MD_P10_DROP
                        }
                    }
                    
                    # Deduplicate: per-check-type deduplication (independent tracking)
                    # Filter out breaches that were recently alerted
                    breaches_to_alert = []
                    for breach in breaches:
                        check_type = breach["check"]
                        if check_type in last_alert_time:
                            time_since = time.time() - last_alert_time[check_type]
                            if time_since < ALERT_INTERVAL_SECONDS:
                                logging.debug(f"Skipping {check_type} alert (sent {time_since:.0f}s ago, < {ALERT_INTERVAL_SECONDS}s)")
                                continue
                        breaches_to_alert.append(breach)
                    
                    if breaches_to_alert:
                        # Update payload with filtered breaches
                        alert_payload["breaches"] = breaches_to_alert
                        
                        # Send alerts
                        alert_sent = False
                        
                        if ALERT_WEBHOOK_URL:
                            if send_webhook_alert(ALERT_WEBHOOK_URL, alert_payload):
                                logging.info(f"✓ Webhook alert sent: {len(breaches_to_alert)} breach(es)")
                                alert_sent = True
                        
                        if ALERT_EMAIL_TO:
                            subject = f"Docling QA Alert: {len(breaches_to_alert)} Quality Breach(es)"
                            body = json.dumps(alert_payload, indent=2)
                            if send_email_alert(ALERT_EMAIL_TO, subject, body):
                                logging.info(f"✓ Email alert sent to {ALERT_EMAIL_TO}")
                                alert_sent = True
                        
                        if alert_sent:
                            # Update last alert time for each check type that was alerted
                            for breach in breaches_to_alert:
                                last_alert_time[breach["check"]] = time.time()
                        else:
                            logging.warning("Alert configured but failed to send")
                    else:
                        logging.info(f"All breaches were recently alerted (deduplication)")
                else:
                    logging.debug("No breaches detected")
                
                # Sleep until next check
                time.sleep(ALERT_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                logging.info("Stopping alerts watch...")
                break
            except Exception as e:
                logging.error(f"Error in alert check: {e}", exc_info=True)
                time.sleep(ALERT_INTERVAL_SECONDS)
    
    except Exception as e:
        logging.error(f"Failed to connect to MongoDB: {e}")
        sys.exit(1)
    finally:
        if 'client' in locals():
            client.close()


if __name__ == "__main__":
    main()

