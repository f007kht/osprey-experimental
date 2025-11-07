"""Application configuration and settings.

Module 1 - Osprey Backend: Document processing application.
"""

import os
from functools import lru_cache

# Import app_settings for backward compatibility
from app_settings import get_settings as _get_app_settings


@lru_cache
def get_settings():
    """Get application settings (wraps app_settings.get_settings for now).
    
    Returns:
        AppSettings instance
    """
    return _get_app_settings()


# QA feature flags (read once at startup)
QA_FLAG_ENABLE_PDF_WARNING_SUPPRESS = os.getenv("QA_FLAG_ENABLE_PDF_WARNING_SUPPRESS", "1") == "1"
QA_FLAG_ENABLE_TEXT_LAYER_DETECT = os.getenv("QA_FLAG_ENABLE_TEXT_LAYER_DETECT", "1") == "1"
QA_FLAG_LOG_NORMALIZED_CODES = os.getenv("QA_FLAG_LOG_NORMALIZED_CODES", "1") == "1"
QA_SCHEMA_VERSION = int(os.getenv("QA_SCHEMA_VERSION", "2"))

# Guardrails (configurable via env, with safe defaults)
MAX_PAGES = int(os.getenv("QA_MAX_PAGES", "500"))
MAX_SECONDS = float(os.getenv("QA_MAX_SECONDS", "300"))

