"""
Logging configuration for the multilingual emotion detection system.

This module configures logging for the system, providing consistent
logging formats and handlers across all modules.
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Dict, Any, Union

from .config import get_config

# Get configuration
config = get_config()


def setup_logger(
    name: Optional[str] = None,
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None
) -> logging.Logger:
    """
    Set up and configure a logger with the specified parameters.
    
    Args:
        name: Logger name, typically the module name (__name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to the log file
        log_format: Log message format
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    # Use provided values or fall back to configuration
    name = name or __name__
    level = level or config.get("LOGGING", "LEVEL", "INFO")
    log_file = log_file or config.get("LOGGING", "FILE_PATH")
    log_format = log_format or config.get("LOGGING", "FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    max_bytes = max_bytes or config.get("LOGGING", "MAX_SIZE", 10485760)  # 10 MB default
    backup_count = backup_count or config.get("LOGGING", "BACKUP_COUNT", 5)
    
    # Create logger
    logger = logging.getLogger(name)
    
    # Convert string level to logging level constant
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO
            print(f"Invalid log level: {level}, using INFO")
        level = numeric_level
        
    logger.setLevel(level)
    
    # Don't add handlers if they already exist (prevents duplicate logs)
    if logger.handlers:
        return logger
        
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file path is provided
    if log_file:
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(log_file)
            os.makedirs(log_dir, exist_ok=True)
            
            # Add rotating file handler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Could not set up log file: {str(e)}")
    
    return logger


# Create default logger
logger = setup_logger()


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a configured logger for the specified name.
    
    Args:
        name: Logger name, typically the module name (__name__)
        
    Returns:
        Configured logger instance
    """
    return setup_logger(name=name)


def set_log_level(level: Union[str, int]) -> None:
    """
    Set the log level for all loggers in the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string level to logging level constant if needed
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            print(f"Invalid log level: {level}, using INFO")
            numeric_level = logging.INFO
        level = numeric_level
    
    # Update root logger
    logging.getLogger().setLevel(level)
    
    # Update application's main logger
    logger.setLevel(level)
    
    # Update config
    config.update("LOGGING", "LEVEL", logging.getLevelName(level))

