#!/usr/bin/env python3
"""
Backend-only test server for Sprint 6 verification.
"""

import sys
from pathlib import Path
import uvicorn

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.api import app
from app.logging_conf import setup_logging

if __name__ == "__main__":
    setup_logging(debug=True)
    
    # Run just the FastAPI backend
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )