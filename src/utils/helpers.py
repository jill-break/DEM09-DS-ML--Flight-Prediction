"""
Helper Utilities Module

This module contains reusable helper functions used across the project.
Functions here should be stateless, pure, and general-purpose.

Design Decision: Centralizing common utilities prevents code duplication
and ensures consistent behavior across modules.
"""

import json
import joblib
from pathlib import Path
from typing import Any, Dict
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_joblib(obj: Any, filepath: Path) -> None:
    """
    Serialize and save a Python object to disk using joblib.
    
    Joblib is optimized for large numpy arrays and scikit-learn models,
    providing better performance than pickle for ML artifacts.
    
    Args:
        obj: Python object to serialize
        filepath: Destination file path
    
    Raises:
        IOError: If file cannot be written
    """
    try:
        joblib.dump(obj, filepath)
        logger.info(f"Object successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save object to {filepath}: {str(e)}")
        raise


def load_joblib(filepath: Path) -> Any:
    """
    Load and deserialize a joblib object from disk.
    
    Args:
        filepath: Path to the joblib file
    
    Returns:
        Deserialized Python object
    
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    try:
        obj = joblib.load(filepath)
        logger.info(f"Object successfully loaded from {filepath}")
        return obj
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Failed to load object from {filepath}: {str(e)}")
        raise


# Backward-compatible aliases
save_pickle = save_joblib
load_pickle = load_joblib


def save_json(data: Dict, filepath: Path, indent: int = 4) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Destination file path
        indent: JSON indentation level for readability
    
    Raises:
        IOError: If file cannot be written
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        logger.info(f"JSON data successfully saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {str(e)}")
        raise


def load_json(filepath: Path) -> Dict:
    """
    Load a dictionary from a JSON file.
    
    Args:
        filepath: Path to the JSON file
    
    Returns:
        Dictionary loaded from JSON
    
    Raises:
        FileNotFoundError: If file doesn't exist
        JSONDecodeError: If JSON is invalid
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"JSON data successfully loaded from {filepath}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {str(e)}")
        raise


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across libraries.
    
    Args:
        seed: Random seed value
    
    Design Decision: Setting seeds for numpy and Python's random module
    ensures reproducible results in data splitting, model initialization,
    and stochastic algorithms.
    """
    np.random.seed(seed)
    import random
    random.seed(seed)
    logger.info(f"Random seed set to {seed} for reproducibility")


def validate_file_exists(filepath: Path) -> bool:
    """
    Check if a file exists and log the result.
    
    Args:
        filepath: Path to validate
    
    Returns:
        True if file exists, False otherwise
    """
    exists = filepath.exists()
    if exists:
        logger.info(f"File validated: {filepath}")
    else:
        logger.warning(f"File not found: {filepath}")
    return exists