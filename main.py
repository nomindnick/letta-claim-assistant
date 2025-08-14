#!/usr/bin/env python3
"""
Letta Construction Claim Assistant - Main Entry Point

This is the main entry point for the Letta Construction Claim Assistant
desktop application. It sets up the environment and launches the NiceGUI
interface.

Usage:
    python main.py

The application will attempt to run in native desktop mode first,
falling back to browser mode if native mode fails.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.settings import settings
from app.logging_conf import setup_logging, get_logger
from ui.main import main as ui_main


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


def main():
    """Main application entry point."""
    
    # Set up application
    if not setup_application():
        sys.exit(1)
    
    logger = get_logger(__name__)
    
    try:
        # Start the UI
        logger.info("Launching user interface")
        ui_main()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error("Application failed", error=str(e))
        sys.exit(1)
    finally:
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    main()