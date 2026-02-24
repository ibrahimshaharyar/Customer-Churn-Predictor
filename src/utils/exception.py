"""
Custom Exception Handler for Banking Customer Churn Pipeline

This module provides a custom exception class that captures detailed error information
including the file name, line number, and error message for easy debugging.
"""

import sys
from src.utils.logger import logger


def get_error_details(error, error_detail: sys):
    """
    Extract detailed error information from exception
    
    Args:
        error: The exception object
        error_detail: sys module to extract traceback
        
    Returns:
        str: Formatted error message with file, line number, and error details
    """
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the file name where error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Get the line number where error occurred
    line_number = exc_tb.tb_lineno
    
    # Format error message
    error_message = f"""
    ╔════════════════════════════════════════════════════════════════╗
    ║                   CUSTOM EXCEPTION RAISED                      ║
    ╠════════════════════════════════════════════════════════════════╣
    ║ Error occurred in: {file_name}
    ║ Line number: {line_number}
    ║ Error message: {str(error)}
    ╚════════════════════════════════════════════════════════════════╝
    """
    
    return error_message


class CustomException(Exception):
    """
    Custom exception class for the churn prediction pipeline
    
    This exception captures:
    - The original error message
    - The file where the error occurred
    - The line number of the error
    - Full traceback for debugging
    
    Usage:
        try:
            # Your code here
            risky_operation()
        except Exception as e:
            logger.error("Operation failed")
            raise CustomException(e, sys)
    """
    
    def __init__(self, error_message, error_detail: sys):
        """
        Initialize custom exception
        
        Args:
            error_message: The original error message or Exception object
            error_detail: sys module to extract traceback information
        """
        super().__init__(error_message)
        
        # Get detailed error information
        self.error_message = get_error_details(error_message, error_detail)
        
        # Log the error
        logger.error(self.error_message)
    
    def __str__(self):
        """Return formatted error message"""
        return self.error_message
