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
import multiprocessing

# Set multiprocessing start method to 'spawn' to avoid fork/spawn context issues
# This must be done before any other imports that might use multiprocessing
if __name__ in {"__main__", "__mp_main__"}:
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.settings import settings
from app.logging_conf import setup_logging, get_logger
from app.api import app as fastapi_app
from ui.main import create_ui_app
from app.letta_server import server_manager
from app.letta_config import config_manager


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
        
        # Initialize Letta server if configured
        if config.letta_server_auto_start:
            logger.info("Initializing Letta server")
            try:
                # Configure server manager from settings
                server_manager.configure(
                    mode=config.letta_server_mode,
                    host=config.letta_server_host,
                    port=config.letta_server_port,
                    startup_timeout=config.letta_server_startup_timeout,
                    health_check_interval=config.letta_server_health_check_interval
                )
                
                # Start server
                if server_manager.start():
                    logger.info(f"Letta server started on {server_manager.get_base_url()}")
                else:
                    logger.warning("Letta server failed to start, continuing in fallback mode")
            except Exception as e:
                logger.error(f"Failed to initialize Letta server: {e}")
                logger.warning("Continuing without Letta server (fallback mode)")
        
        logger.info("Application setup completed successfully")
        return True
        
    except Exception as e:
        print(f"Failed to set up application: {e}")
        return False


def get_available_port(start_port=8000, max_attempts=100):
    """Find an available port starting from start_port."""
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")


def start_backend_server():
    """Start the FastAPI backend server in a separate thread."""
    logger = get_logger(__name__)
    logger.info("Starting FastAPI backend server")
    
    # Find an available port
    port = get_available_port(8000)
    
    if port != 8000:
        logger.info(f"Port 8000 is in use, using port {port} instead")
    
    # Configure uvicorn for embedded use
    config = uvicorn.Config(
        fastapi_app,
        host="127.0.0.1",
        port=port,
        log_level="info",
        access_log=False  # Reduce noise in logs
    )
    
    # Store the port for the UI to use
    import os
    os.environ['BACKEND_PORT'] = str(port)
    
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
        # Clean shutdown of Letta server
        try:
            if server_manager._is_running:
                logger.info("Stopping Letta server")
                server_manager.stop()
        except Exception as e:
            logger.error(f"Error stopping Letta server: {e}")
        
        logger.info("Application shutdown complete")


if __name__ in {"__main__", "__mp_main__"}:
    main()