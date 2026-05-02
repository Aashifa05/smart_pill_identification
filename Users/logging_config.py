"""
Pill System Logging Configuration Module
==========================================

This module provides a centralized logging system for monitoring pill identification events.
It handles all system events including image uploads, validation, preprocessing, and predictions.

Events logged:
1. Image upload request received
2. File type validation (valid or invalid image format)
3. Corrupted or unreadable image detection
4. Image preprocessing started
5. Model prediction started
6. Model prediction completed with predicted pill class
7. Runtime errors or exceptions

Log file: pill_system.log
"""

import logging
import os
from pathlib import Path

# Get logger for pill system events
def get_pill_logger():
    """
    Returns the pill_system logger configured in Django settings.
    
    Returns:
        logging.Logger: The pill_system logger
    """
    return logging.getLogger('pill_system')


# ==================== IMAGE UPLOAD EVENTS ====================

def log_image_upload(filename, file_size, user_info=None):
    """
    Log when an image upload request is received.
    
    Args:
        filename (str): Name of the uploaded file
        file_size (int): Size of the file in bytes
        user_info (str, optional): Additional user context information
    """
    logger = get_pill_logger()
    message = f"Image uploaded successfully - File: {filename}, Size: {file_size} bytes"
    if user_info:
        message += f", User: {user_info}"
    logger.info(message)


# ==================== FILE VALIDATION EVENTS ====================

def log_valid_file_type(filename, file_type):
    """
    Log when a file passes type validation.
    
    Args:
        filename (str): Name of the file
        file_type (str): File type/extension (e.g., 'jpeg', 'png')
    """
    logger = get_pill_logger()
    logger.info(f"File type validation passed - File: {filename}, Type: {file_type}")


def log_invalid_file_type(filename, file_type, allowed_types=None):
    """
    Log when a file fails type validation.
    
    Args:
        filename (str): Name of the file
        file_type (str): File type/extension that was received
        allowed_types (list, optional): List of allowed file types
    """
    logger = get_pill_logger()
    message = f"Invalid file type uploaded - File: {filename}, Type: {file_type}"
    if allowed_types:
        message += f", Allowed: {', '.join(allowed_types)}"
    logger.warning(message)


# ==================== IMAGE CORRUPTION DETECTION ====================

def log_corrupted_image(filename, error_details=None):
    """
    Log when a corrupted or unreadable image is detected.
    
    Args:
        filename (str): Name of the problematic file
        error_details (str, optional): Details about why the image is corrupted
    """
    logger = get_pill_logger()
    message = f"Corrupted image detected - File: {filename}"
    if error_details:
        message += f", Error: {error_details}"
    logger.error(message)


# ==================== IMAGE PREPROCESSING EVENTS ====================

def log_preprocessing_started(filename):
    """
    Log when image preprocessing starts.
    
    Args:
        filename (str): Name of the file being preprocessed
    """
    logger = get_pill_logger()
    logger.info(f"Image preprocessing started - File: {filename}")


def log_preprocessing_completed(filename, preprocessing_steps=None):
    """
    Log when image preprocessing is completed.
    
    Args:
        filename (str): Name of the file
        preprocessing_steps (list, optional): List of preprocessing steps applied
    """
    logger = get_pill_logger()
    message = f"Image preprocessing completed - File: {filename}"
    if preprocessing_steps:
        message += f", Steps: {', '.join(preprocessing_steps)}"
    logger.info(message)


# ==================== MODEL PREDICTION EVENTS ====================

def log_prediction_started(filename, model_name=None):
    """
    Log when model prediction starts.
    
    Args:
        filename (str): Name of the file being processed
        model_name (str, optional): Name of the model being used
    """
    logger = get_pill_logger()
    message = f"Model prediction started - File: {filename}"
    if model_name:
        message += f", Model: {model_name}"
    logger.info(message)


def log_prediction_completed(pill_name, confidence, model_details=None):
    """
    Log when model prediction is completed.
    
    Args:
        pill_name (str): Name of the predicted pill
        confidence (float or str): Confidence score of the prediction
        model_details (dict, optional): Additional model prediction details
    """
    logger = get_pill_logger()
    message = f"Prediction completed: {pill_name} (Confidence: {confidence})"
    if model_details:
        if 'top_3' in model_details:
            message += f", Top 3: {model_details['top_3']}"
        if 'processing_time' in model_details:
            message += f", Processing time: {model_details['processing_time']}ms"
    logger.info(message)


def log_low_confidence_prediction(pill_name, confidence, threshold):
    """
    Log when a prediction has low confidence below threshold.
    
    Args:
        pill_name (str): Name of the predicted pill
        confidence (float or str): Actual confidence score
        threshold (float or str): Confidence threshold that was not met
    """
    logger = get_pill_logger()
    logger.warning(
        f"Low confidence prediction - Pill: {pill_name}, "
        f"Confidence: {confidence}, Below threshold: {threshold}"
    )


# ==================== ERROR LOGGING ====================

def log_system_error(error_message, error_type=None, stack_trace=None):
    """
    Log runtime errors or exceptions.
    
    Args:
        error_message (str): Description of the error
        error_type (str, optional): Type of error (e.g., 'FileNotFoundError', 'ModelLoadError')
        stack_trace (str, optional): Full stack trace if available
    """
    logger = get_pill_logger()
    message = f"System error: {error_message}"
    if error_type:
        message = f"System error ({error_type}): {error_message}"
    if stack_trace:
        logger.error(message + f"\nStack trace:\n{stack_trace}")
    else:
        logger.error(message)


def log_file_handling_error(filename, operation, error_details):
    """
    Log errors related to file handling.
    
    Args:
        filename (str): Name of the file
        operation (str): Operation that failed (e.g., 'read', 'write', 'delete')
        error_details (str): Details about the error
    """
    logger = get_pill_logger()
    logger.error(f"File handling error - File: {filename}, Operation: {operation}, Error: {error_details}")


def log_model_error(error_message, model_name=None):
    """
    Log errors related to model operations.
    
    Args:
        error_message (str): Description of the model error
        model_name (str, optional): Name of the model that failed
    """
    logger = get_pill_logger()
    message = f"Model error: {error_message}"
    if model_name:
        message = f"Model error ({model_name}): {error_message}"
    logger.error(message)


# ==================== DATABASE EVENTS ====================

def log_database_write(operation, table_name, record_count=1):
    """
    Log database write operations.
    
    Args:
        operation (str): Type of operation (e.g., 'INSERT', 'UPDATE', 'DELETE')
        table_name (str): Name of the affected table
        record_count (int): Number of records affected
    """
    logger = get_pill_logger()
    logger.info(f"Database {operation} - Table: {table_name}, Records: {record_count}")


def log_database_error(error_message, table_name=None):
    """
    Log database errors.
    
    Args:
        error_message (str): Description of the database error
        table_name (str, optional): Name of the affected table
    """
    logger = get_pill_logger()
    message = f"Database error: {error_message}"
    if table_name:
        message = f"Database error (Table: {table_name}): {error_message}"
    logger.error(message)


# ==================== USER ACTIVITY LOGGING ====================

def log_user_action(action, user_info=None, details=None):
    """
    Log user actions in the system.
    
    Args:
        action (str): Type of action (e.g., 'login', 'logout', 'prediction_request')
        user_info (str, optional): User identification information
        details (str, optional): Additional details about the action
    """
    logger = get_pill_logger()
    message = f"User action: {action}"
    if user_info:
        message += f" - User: {user_info}"
    if details:
        message += f" - Details: {details}"
    logger.info(message)


# ==================== SAFETY AND VALIDATION ====================

def log_safety_validation(passed, pill_name, validation_details=None):
    """
    Log medical safety validation results.
    
    Args:
        passed (bool): Whether the validation passed
        pill_name (str): Name of the pill being validated
        validation_details (dict, optional): Details about validation checks
    """
    logger = get_pill_logger()
    status = "PASSED" if passed else "FAILED"
    message = f"Safety validation {status} - Pill: {pill_name}"
    if validation_details:
        message += f", Details: {validation_details}"
    log_level = logger.info if passed else logger.warning
    log_level(message)


def log_csv_write(filename, record_info=None):
    """
    Log CSV file write operations.
    
    Args:
        filename (str): Name of the CSV file
        record_info (dict, optional): Information about the record being written
    """
    logger = get_pill_logger()
    message = f"CSV record written - File: {filename}"
    if record_info and 'pill_name' in record_info:
        message += f", Pill: {record_info['pill_name']}"
    logger.info(message)


# ==================== DEBUG LOGGING ====================

def log_debug(message, context=None):
    """
    Log debug information.
    
    Args:
        message (str): Debug message
        context (dict, optional): Additional context information
    """
    logger = get_pill_logger()
    if context:
        message += f" - Context: {context}"
    logger.debug(message)


# ==================== SYSTEM STATUS ====================

def log_system_startup():
    """Log system startup."""
    logger = get_pill_logger()
    logger.info("=" * 80)
    logger.info("Pill Identification System Started")
    logger.info("=" * 80)


def log_system_shutdown():
    """Log system shutdown."""
    logger = get_pill_logger()
    logger.info("=" * 80)
    logger.info("Pill Identification System Shutdown")
    logger.info("=" * 80)
