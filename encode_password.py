#!/usr/bin/env python3
"""
Secure password URL encoder for MongoDB connection strings.

Uses getpass to prevent password from appearing in terminal history.
"""

import getpass
import urllib.parse
import sys

def main():
    print("MongoDB Password URL Encoder", file=sys.stderr)
    print("=" * 40, file=sys.stderr)
    print("Enter your MongoDB password (input will be hidden):", file=sys.stderr)
    
    password = getpass.getpass("Password: ")
    
    if not password:
        print("ERROR: Empty password provided", file=sys.stderr)
        sys.exit(1)
    
    encoded = urllib.parse.quote_plus(password)
    
    # Only print the encoded password (no echo of original)
    print(encoded)
    
    # Show confirmation (without revealing password)
    print(f"\n✓ Password encoded successfully (length: {len(encoded)} chars)", file=sys.stderr)
    print("Copy the encoded password above and use it in your connection string.", file=sys.stderr)

if __name__ == "__main__":
    main()

