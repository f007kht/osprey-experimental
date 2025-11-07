#!/usr/bin/env python3
"""
Helper script to update MongoDB connection string in .env file.

This script:
1. Prompts for MongoDB username, cluster, and password
2. URL-encodes the password securely
3. Creates/updates .env file with the connection string
4. Never prints the full connection string with password
"""

import getpass
import urllib.parse
import os
from pathlib import Path

def main():
    print("MongoDB Credential Update Helper")
    print("=" * 50)
    print()
    
    # Get MongoDB connection details
    print("Enter your MongoDB Atlas connection details:")
    username = input("Username: ").strip()
    cluster = input("Cluster hostname (e.g., cluster0.xxxxx.mongodb.net): ").strip()
    
    print("\nEnter your MongoDB password (input will be hidden):")
    password = getpass.getpass("Password: ")
    
    if not all([username, cluster, password]):
        print("\nERROR: All fields are required", file=os.sys.stderr)
        os.sys.exit(1)
    
    # URL-encode the password
    encoded_password = urllib.parse.quote_plus(password)
    
    # Build connection string (without printing it)
    connection_string = (
        f"mongodb+srv://{username}:{encoded_password}@"
        f"{cluster}/?retryWrites=true&w=majority&appName=osprey"
    )
    
    # Read existing .env or create new
    env_path = Path(".env")
    env_lines = []
    mongodb_found = False
    
    if env_path.exists():
        with open(env_path, "r") as f:
            env_lines = f.readlines()
        
        # Check if MONGODB_CONNECTION_STRING already exists
        for i, line in enumerate(env_lines):
            if line.strip().startswith("MONGODB_CONNECTION_STRING="):
                env_lines[i] = f"MONGODB_CONNECTION_STRING={connection_string}\n"
                mongodb_found = True
                break
    
    # Add if not found
    if not mongodb_found:
        env_lines.append(f"MONGODB_CONNECTION_STRING={connection_string}\n")
    
    # Ensure ENABLE_MONGODB is set
    mongodb_enabled_found = False
    for i, line in enumerate(env_lines):
        if line.strip().startswith("ENABLE_MONGODB="):
            mongodb_enabled_found = True
            # Ensure it's set to true
            if "true" not in line.lower():
                env_lines[i] = "ENABLE_MONGODB=true\n"
            break
    
    if not mongodb_enabled_found:
        env_lines.append("ENABLE_MONGODB=true\n")
    
    # Write .env file
    with open(env_path, "w") as f:
        f.writelines(env_lines)
    
    print(f"\n✓ .env file updated successfully")
    print(f"  - Username: {username}")
    print(f"  - Cluster: {cluster}")
    print(f"  - Password: [ENCODED - {len(encoded_password)} chars]")
    print(f"\nConnection string has been saved to .env file.")
    print("The password is URL-encoded and will not be displayed.")
    print("\nNext steps:")
    print("1. Run: python quick_check_mongo.py")
    print("2. Restart your Streamlit app to load the new credentials")

if __name__ == "__main__":
    main()

