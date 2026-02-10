"""
Logging Utility Module

This module provides centralized logging configuration for the entire project.
It ensures consistent log formatting, handles both console and file logging,
and prevents duplicate log entries.

Design Decision: Using Python's built-in logging module ensures thread-safety,
flexibility, and production-grade logging capabilities.
"""

import logging
import sys
from pathlib import Path
from src.config import config


def setup_logger(name: str, log_file: Path = config.LOG_FILE, 
                 level: str = config.LOG_LEVEL) -> logging.Logger:
    """
    Configure and return a logger instance with console and file handlers.
    
    Args:
        name: Name of the logger (typically __name__ of the calling module)
        log_file: Path to the log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    
    Design Pattern: Factory pattern for logger creation ensures consistency
    across all modules while preventing duplicate handlers.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger for the specified module.
    
    Args:
        name: Name of the logger (typically __name__)
    
    Returns:
        Logger instance
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Processing started")
    """
    return setup_logger(name)