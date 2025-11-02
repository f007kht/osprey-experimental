# Streamlit startup script with MongoDB enabled
$env:ENABLE_MONGODB="true"
$env:MONGODB_CONNECTION_STRING="mongodb+srv://jlcaraveo1_db_user:Ey6LsnaPhRvWu14X@cluster0.tobyztw.mongodb.net/?appName=Cluster0"
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
$env:LC_ALL="en_US.UTF-8"

Write-Host "Starting Streamlit with MongoDB enabled..."
Write-Host "ENABLE_MONGODB=$env:ENABLE_MONGODB"
Write-Host "MongoDB connection string is set: $($env:MONGODB_CONNECTION_STRING -ne '')"

streamlit run app.py

