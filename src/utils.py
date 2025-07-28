import time
import logging
from typing import Any, Dict, List
from pathlib import Path

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def measure_time(func):
    """Decorator to measure function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  {func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def ensure_directory_exists(path: Path):
    """Ensure directory exists with proper permissions"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_filename(text: str, max_length: int = 50) -> str:
    """Create safe filename from text"""
    # Remove unsafe characters
    safe_text = "".join(c for c in text if c.isalnum() or c in (' ', '-', '_')).rstrip()
    # Truncate if too long
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length].rstrip()
    return safe_text.replace(' ', '_')
