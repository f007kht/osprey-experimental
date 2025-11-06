# MongoDB Credential Rotation Guide

This guide helps you safely rotate MongoDB credentials without exposing passwords in logs.

## Step 1: Encode Your Password

Run the password encoder (password input is hidden):

```bash
python encode_password.py
```

**OR** use the interactive helper that updates `.env` automatically:

```bash
python update_mongodb_credentials.py
```

The helper will:
- Prompt for username, cluster, and password
- URL-encode the password securely
- Create/update `.env` file
- Never display the full connection string

## Step 2: Verify Connection

Test the connection with the new credentials:

```bash
python quick_check_mongo.py
```

Expected output: `{'ok': 1.0}`

If you get an error, check:
- Password is correctly URL-encoded (special characters like `@:/#?&%` must be encoded)
- Username and cluster hostname are correct
- IP whitelist in MongoDB Atlas includes your current IP

## Step 3: Restart Application

**For local Streamlit:**
```powershell
# Stop current process (Ctrl+C if running)
# Then restart:
streamlit run app.py
```

**For Docker:**
```bash
docker compose down
docker compose up -d --build
```

## Step 4: Verify After Restart

Run the connection test again:

```bash
python quick_check_mongo.py
```

## Manual .env Update

If you prefer to update `.env` manually:

1. Encode password:
   ```bash
   python encode_password.py
   # Copy the encoded password output
   ```

2. Edit `.env` file:
   ```bash
   MONGODB_CONNECTION_STRING=mongodb+srv://<USERNAME>:<ENCODED_PASSWORD>@<cluster>.mongodb.net/?retryWrites=true&w=majority&appName=osprey
   ENABLE_MONGODB=true
   ```

3. Replace:
   - `<USERNAME>` with your MongoDB username
   - `<ENCODED_PASSWORD>` with the encoded password from step 1
   - `<cluster>` with your cluster hostname

## Security Notes

- `.env` is already in `.gitignore` - never commit it
- Passwords are URL-encoded to handle special characters
- Connection strings are masked in logs (password shown as `***`)
- Never share or log the full connection string

