"""
Logging configuration for structured logging with matter context.

Provides structured logging setup, file rotation, and debug/production modes.
All logs include matter-specific context when available.
"""

import logging
import structlog
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    matter_path: Optional[Path] = None,
    debug: bool = False,
    log_to_file: bool = True
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        matter_path: Path to current matter for context logging
        debug: Enable debug-level logging
        log_to_file: Whether to log to file in addition to console
    """
    
    # Set logging level
    log_level = logging.DEBUG if debug else logging.INFO
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer(colors=True)
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Add file logging if requested and matter_path provided
    if log_to_file and matter_path:
        log_dir = matter_path / "logs"
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / "app.log"
        
        # Add file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create formatter for file output
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def bind_matter_context(logger: structlog.stdlib.BoundLogger, matter_id: str, matter_name: str) -> structlog.stdlib.BoundLogger:
    """
    Bind matter context to a logger instance.
    
    Args:
        logger: Logger to bind context to
        matter_id: Matter ID
        matter_name: Matter name
    
    Returns:
        Logger with bound matter context
    """
    return logger.bind(matter_id=matter_id, matter_name=matter_name)


# Pre-configured logger for general use
logger = get_logger(__name__)