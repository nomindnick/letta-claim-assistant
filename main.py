#!/usr/bin/env python3
"""
Letta Construction Claim Assistant - Main Entry Point

This is the main entry point for the Letta Construction Claim Assistant
desktop application. It sets up the environment and launches the NiceGUI
interface with integrated FastAPI backend.

Usage:
    python main.py

The application will attempt to run in native desktop mode first,
falling back to browser mode if native mode fails.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional
import threading
import uvicorn

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.settings import settings
from app.logging_conf import setup_logging, get_logger
from app.api import app as fastapi_app
from ui.main import create_ui_app


def setup_application() -> bool:
    """
    Set up the application environment.
    
    Returns:
        True if setup was successful, False otherwise
    """
    try:
        # Set up logging
        setup_logging(debug=True)
        logger = get_logger(__name__)
        
        logger.info("Starting Letta Construction Claim Assistant")
        logger.info("Application setup initiated")
        
        # Validate configuration
        config = settings.global_config
        logger.info(
            "Configuration loaded", 
            provider=config.llm_provider,
            model=config.llm_model,
            data_root=str(config.data_root)
        )
        
        # Ensure data directory exists
        config.data_root.mkdir(parents=True, exist_ok=True)
        
        logger.info("Application setup completed successfully")
        return True
        
    except Exception as e:
        print(f"Failed to set up application: {e}")
        return False


def start_backend_server():
    """Start the FastAPI backend server in a separate thread."""
    logger = get_logger(__name__)
    logger.info("Starting FastAPI backend server")
    
    # Configure uvicorn for embedded use
    config = uvicorn.Config(
        fastapi_app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
        access_log=False  # Reduce noise in logs
    )
    
    server = uvicorn.Server(config)
    server.run()


def main():
    """Main application entry point."""
    
    # Set up application
    if not setup_application():
        sys.exit(1)
    
    logger = get_logger(__name__)
    
    try:
        # Start FastAPI backend in background thread
        logger.info("Starting backend server in background")
        backend_thread = threading.Thread(
            target=start_backend_server,
            daemon=True,
            name="FastAPI-Backend"
        )
        backend_thread.start()
        
        # Give backend a moment to start
        import time
        time.sleep(2)
        
        # Start the UI
        logger.info("Launching user interface")
        create_ui_app()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Application failed", error=str(e))
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")


if __name__ in {"__main__", "__mp_main__"}:
    main()