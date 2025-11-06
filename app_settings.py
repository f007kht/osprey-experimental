"""
Application settings module using pydantic-settings for centralized environment variable management.

This module provides a single source of truth for all application configuration,
loading from environment variables with validation and sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    # Fallback for environments without pydantic-settings
    # Use basic class with os.getenv() if pydantic-settings not available
    BaseSettings = object
    SettingsConfigDict = None
    Field = None
    field_validator = None
    PYDANTIC_AVAILABLE = False


class AppSettings:
    """
    Application settings loaded from environment variables.
    
    Supports loading from .env file (if python-dotenv is available) and
    environment variables, with validation and sensible defaults.
    """
    
    def __init__(self):
        """Initialize settings from environment variables."""
        # Try to load from .env file if python-dotenv is available
        try:
            from dotenv import load_dotenv
            # Load .env file from project root (if it exists)
            env_path = Path(__file__).parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            pass  # python-dotenv not available, skip .env loading
        
        # Load settings from environment variables
        self.enable_downloads = self._get_bool("ENABLE_DOWNLOADS", True)
        self.enable_mongodb = self._get_bool("ENABLE_MONGODB", False)
        
        self.mongodb_connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        self.mongodb_database = os.getenv("MONGODB_DATABASE", "docling_documents")
        self.mongodb_collection = os.getenv("MONGODB_COLLECTION", "documents")
        
        self.page_batch_size = int(os.getenv("PAGE_BATCH_SIZE", "1"))
        if self.page_batch_size < 1 or self.page_batch_size > 10:
            self.page_batch_size = 1  # Clamp to valid range
        
        # Image generation controls
        self.images_enable = self._get_bool("IMAGES_ENABLE", True)
        self.images_scale = float(os.getenv("IMAGES_SCALE", "0.75"))
        # IMAGES_SCALE bounds (tighten to 0.0-1.0; clamp with conservative default)
        if self.images_scale < 0.0 or self.images_scale > 1.0:
            self.images_scale = 0.75  # Clamp to valid range
        
        self.pipeline_queue_max = int(os.getenv("PIPELINE_QUEUE_MAX", "3"))
        # PIPELINE_QUEUE_MAX bounds (tighten to 1-4; clamp with conservative default)
        if self.pipeline_queue_max < 1 or self.pipeline_queue_max > 4:
            self.pipeline_queue_max = 3  # Clamp to valid range
        
        # Optional: ensure not exceeding available cores (best-effort, won't crash)
        try:
            cpu_count = os.cpu_count() or 1
            if self.pipeline_queue_max > max(1, min(4, cpu_count)):
                self.pipeline_queue_max = max(1, min(4, cpu_count))
        except Exception:
            pass
        
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.use_remote_embeddings = self._get_bool("USE_REMOTE_EMBEDDINGS", False)
        self.voyageai_api_key = os.getenv("VOYAGEAI_API_KEY")
        
        # Port: Use $PORT if available (for platforms like Spaces, Render, Fly, Heroku)
        self.port = int(os.getenv("PORT", os.getenv("STREAMLIT_SERVER_PORT", "8501")))
        
        # TESSDATA_PREFIX: Auto-detect if not set (will be done in app.py)
        self.tessdata_prefix = os.getenv("TESSDATA_PREFIX")
        
        # OpenCV setting
        self.opencv_io_enable_openexr = os.getenv("OPENCV_IO_ENABLE_OPENEXR", "0")
        
        # Encoding settings
        self.python_io_encoding = os.getenv("PYTHONIOENCODING", "utf-8")
        self.python_utf8 = os.getenv("PYTHONUTF8", "1")
        self.lc_all = os.getenv("LC_ALL", "C.UTF-8")
        
        # MongoDB connection status (set after validation)
        self.mongo_status = "disabled"
    
    @staticmethod
    def _get_bool(env_var: str, default: bool) -> bool:
        """Parse boolean environment variable."""
        value = os.getenv(env_var, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        elif value in ("false", "0", "no", "off", ""):
            return False
        return default
    
    def validate_mongodb_config(self) -> tuple[bool, Optional[str]]:
        """
        Validate MongoDB configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.enable_mongodb:
            return True, None
        
        if not self.mongodb_connection_string:
            return False, "MongoDB connection string is required when ENABLE_MONGODB=true"
        
        if not (self.mongodb_connection_string.startswith("mongodb+srv://") or 
                self.mongodb_connection_string.startswith("mongodb://")):
            return False, "Invalid MongoDB connection string format. Use mongodb+srv://... or mongodb://..."
        
        if self.use_remote_embeddings and not self.voyageai_api_key:
            return False, "VOYAGEAI_API_KEY is required when USE_REMOTE_EMBEDDINGS=true"
        
        return True, None
    
    @staticmethod
    def mask_connection_string(uri: str) -> str:
        """
        Mask password in MongoDB connection string for safe logging.
        
        Args:
            uri: MongoDB connection string
            
        Returns:
            Connection string with password masked as ***
        """
        if not uri:
            return uri
        
        try:
            # Split at :// to get scheme
            if "://" not in uri:
                return uri
            
            scheme, rest = uri.split("://", 1)
            
            # Check if there's a password (format: user:password@host)
            if "@" in rest and ":" in rest.split("@", 1)[0]:
                user_part, tail = rest.split("@", 1)
                if ":" in user_part:
                    username, _ = user_part.split(":", 1)
                    return f"{scheme}://{username}:***@{tail}"
            
            return uri
        except Exception:
            # If parsing fails, return original (better than crashing)
            return uri
    
    @staticmethod
    def safe_mongo_ping(uri: str, timeout_ms: int = 2500) -> tuple[bool, str]:
        """
        Safely ping MongoDB connection without logging secrets.
        
        Args:
            uri: MongoDB connection string
            timeout_ms: Connection timeout in milliseconds
            
        Returns:
            Tuple of (is_connected, status_message)
        """
        from urllib.parse import urlsplit
        
        if not uri:
            return (False, "no_uri")
        
        try:
            from pymongo import MongoClient
            import certifi
            
            client = MongoClient(
                uri,
                serverSelectionTimeoutMS=timeout_ms,
                tls=True,
                tlsCAFile=certifi.where(),
                tlsAllowInvalidCertificates=False,
                tlsAllowInvalidHostnames=False,
            )
            client.admin.command("ping")
            
            # Redact connection string for display
            parsed = urlsplit(uri)
            host = parsed.hostname or "unknown"
            # Remove username/password from host display
            if "@" in host:
                host = host.split("@")[-1]
            
            client.close()
            return (True, f"ok:{host}")
        except Exception as e:
            # Return error class name, not full error message (may contain secrets)
            return (False, str(e.__class__.__name__))


# Singleton instance
_settings_instance: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get the singleton settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = AppSettings()
    return _settings_instance

