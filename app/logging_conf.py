"""
Logging configuration for structured logging with matter context.

Provides structured logging setup, file rotation, and debug/production modes.
All logs include matter-specific context when available.
"""

import logging
import logging.handlers
import structlog
import sys
import re
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def setup_logging(
    matter_path: Optional[Path] = None,
    debug: bool = False,
    log_to_file: bool = True,
    enable_rotation: bool = True,
    enable_masking: bool = True,
    production_mode: bool = False
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        matter_path: Path to current matter for context logging
        debug: Enable debug-level logging
        log_to_file: Whether to log to file in addition to console
        enable_rotation: Enable log rotation
        enable_masking: Enable sensitive data masking
        production_mode: Use production logging configuration
    """
    
    # Set logging level
    log_level = logging.DEBUG if debug else logging.INFO
    if production_mode and not debug:
        log_level = logging.WARNING  # Less verbose in production
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )
    
    # Build processor chain
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
    ]
    
    # Add sensitive data masking processor
    if enable_masking:
        processors.append(_mask_sensitive_data)
    
    processors.extend([
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="ISO"),
    ])
    
    # Use different renderers for console vs production
    if production_mode:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Add file logging if requested
    if log_to_file:
        _setup_file_logging(matter_path, log_level, enable_rotation, production_mode)


def _setup_file_logging(
    matter_path: Optional[Path], 
    log_level: int, 
    enable_rotation: bool,
    production_mode: bool
) -> None:
    """Set up file logging with optional rotation."""
    
    # Determine log directory
    if matter_path:
        log_dir = matter_path / "logs"
    else:
        # Global log directory
        log_dir = Path.home() / ".letta-claim" / "logs"
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Main application log
    log_file = log_dir / "app.log"
    
    if enable_rotation:
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
    else:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
    
    file_handler.setLevel(log_level)
    
    # Create formatter for file output
    if production_mode:
        # JSON format for production
        file_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        # Human-readable format for development
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
    
    file_handler.setFormatter(file_formatter)
    
    # Add to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    # Set up separate error log in production
    if production_mode:
        error_log_file = log_dir / "error.log"
        
        if enable_rotation:
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3,
                encoding='utf-8'
            )
        else:
            error_handler = logging.FileHandler(error_log_file, encoding='utf-8')
        
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        root_logger.addHandler(error_handler)


def _mask_sensitive_data(logger, method_name, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Processor to mask sensitive data in log messages."""
    
    # Patterns to mask
    sensitive_patterns = [
        # API keys and tokens
        (r'api[_-]?key["\s]*[:=]["\s]*([a-zA-Z0-9_-]{20,})', r'api_key=***'),
        (r'token["\s]*[:=]["\s]*([a-zA-Z0-9_.-]{20,})', r'token=***'),
        (r'secret["\s]*[:=]["\s]*([a-zA-Z0-9_.-]{20,})', r'secret=***'),
        
        # Credit card numbers
        (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', r'****-****-****-****'),
        
        # SSN patterns
        (r'\b\d{3}-\d{2}-\d{4}\b', r'***-**-****'),
        
        # Email addresses (partial masking)
        (r'([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', r'\1***@\2'),
        
        # Phone numbers
        (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', r'***-***-****'),
        
        # IP addresses (partial masking)
        (r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.)\d{1,3}\b', r'\1***'),
    ]
    
    # Function to mask a string
    def mask_string(text: str) -> str:
        if not isinstance(text, str):
            return text
        
        for pattern, replacement in sensitive_patterns:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    # Recursively mask sensitive data in event dict
    def mask_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        masked = {}
        for key, value in d.items():
            if isinstance(value, str):
                masked[key] = mask_string(value)
            elif isinstance(value, dict):
                masked[key] = mask_dict(value)
            elif isinstance(value, list):
                masked[key] = [mask_string(item) if isinstance(item, str) else item for item in value]
            else:
                masked[key] = value
        return masked
    
    # Mask the main event message
    if 'event' in event_dict and isinstance(event_dict['event'], str):
        event_dict['event'] = mask_string(event_dict['event'])
    
    # Mask other fields recursively
    masked_dict = mask_dict(event_dict)
    
    return masked_dict


def setup_production_logging(
    global_log_dir: Optional[Path] = None,
    enable_rotation: bool = True
) -> None:
    """
    Set up production logging configuration.
    
    Args:
        global_log_dir: Directory for global application logs
        enable_rotation: Enable log rotation
    """
    
    if global_log_dir is None:
        global_log_dir = Path.home() / ".letta-claim" / "logs"
    
    setup_logging(
        matter_path=None,
        debug=False,
        log_to_file=True,
        enable_rotation=enable_rotation,
        enable_masking=True,
        production_mode=True
    )
    
    # Set up additional global logging
    global_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Performance log
    perf_log = global_log_dir / "performance.log"
    perf_handler = logging.handlers.RotatingFileHandler(
        perf_log,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=3,
        encoding='utf-8'
    ) if enable_rotation else logging.FileHandler(perf_log, encoding='utf-8')
    
    perf_handler.setLevel(logging.INFO)
    perf_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "type": "performance", "data": "%(message)s"}'
    )
    perf_handler.setFormatter(perf_formatter)
    
    # Create performance logger
    perf_logger = logging.getLogger("performance")
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    
    # Security log
    security_log = global_log_dir / "security.log"
    security_handler = logging.handlers.RotatingFileHandler(
        security_log,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    ) if enable_rotation else logging.FileHandler(security_log, encoding='utf-8')
    
    security_handler.setLevel(logging.WARNING)
    security_formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "type": "security", "level": "%(levelname)s", "event": "%(message)s"}'
    )
    security_handler.setFormatter(security_formatter)
    
    # Create security logger
    security_logger = logging.getLogger("security")
    security_logger.addHandler(security_handler)
    security_logger.setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def get_performance_logger() -> logging.Logger:
    """Get the performance logger."""
    return logging.getLogger("performance")


def get_security_logger() -> logging.Logger:
    """Get the security logger."""
    return logging.getLogger("security")


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


def log_performance_metric(operation: str, duration_ms: float, **metadata):
    """Log a performance metric."""
    perf_logger = get_performance_logger()
    metric_data = {
        "operation": operation,
        "duration_ms": duration_ms,
        "timestamp": datetime.now().isoformat(),
        **metadata
    }
    perf_logger.info(f"Performance metric: {operation}", extra={"metric": metric_data})


def log_security_event(event_type: str, details: str, **metadata):
    """Log a security-related event."""
    security_logger = get_security_logger()
    event_data = {
        "event_type": event_type,
        "details": details,
        "timestamp": datetime.now().isoformat(),
        **metadata
    }
    security_logger.warning(f"Security event: {event_type}", extra={"security": event_data})


# Pre-configured logger for general use
logger = get_logger(__name__)