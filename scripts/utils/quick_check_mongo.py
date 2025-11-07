#!/usr/bin/env python3
"""
Quick MongoDB connection test script.

Usage:
    MONGODB_CONNECTION_STRING='mongodb+srv://...' python quick_check_mongo.py

Expects: {'ok': 1.0} on success
"""

import os
import sys
import certifi
from pymongo import MongoClient

# Try to load from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not available, skip .env loading

def main():
    uri = os.getenv("MONGODB_CONNECTION_STRING")
    if not uri:
        print("ERROR: MONGODB_CONNECTION_STRING environment variable not set", file=sys.stderr)
        print("Usage: MONGODB_CONNECTION_STRING='mongodb+srv://...' python quick_check_mongo.py", file=sys.stderr)
        sys.exit(1)
    
    try:
        client = MongoClient(
            uri,
            tls=True,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=False,
            tlsAllowInvalidHostnames=False,
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000
        )
        result = client.admin.command("ping")
        client.close()
        print(result)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

