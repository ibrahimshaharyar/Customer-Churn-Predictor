"""
Logging Configuration for Banking Customer Churn Pipeline

This module provides a centralized logging system that logs to both console and file.
All logs are saved to 'logs/churn_pipeline.log' with timestamps and module information.
"""

import logging
import os
from datetime import datetime


# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log file name with timestamp
LOG_FILE = f"churn_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging format
LOG_FORMAT = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str = "churn_pipeline", level=logging.INFO):
    """
    Set up and return a configured logger
    
    Args:
        name: Logger name (default: "churn_pipeline")
        level: Logging level (default: INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger_instance.handlers:
        return logger_instance
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    
    # Console handler (outputs to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # File handler (outputs to log file)
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger_instance.addHandler(console_handler)
    logger_instance.addHandler(file_handler)
    
    return logger_instance


# Create default logger instance
logger = setup_logger()

# Log initial message
logger.info("="*80)
logger.info("Banking Customer Churn Prediction Pipeline - Logging Initialized")
logger.info(f"Log file: {LOG_FILE_PATH}")
logger.info("="*80)


# Example usage patterns for different log levels:
"""
from src.utils.logger import logger

# INFO - General information
logger.info("Starting data preprocessing")

# DEBUG - Detailed debugging information
logger.debug(f"DataFrame shape: {df.shape}")

# WARNING - Warning messages
logger.warning("Missing values detected in column 'Balance'")

# ERROR - Error messages
logger.error("Failed to load model from artifacts")

# CRITICAL - Critical errors
logger.critical("Database connection failed - pipeline cannot continue")
"""
