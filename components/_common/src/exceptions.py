"""Centralized exception handling for the component."""
from typing import Dict, Optional
import re
import asyncio
from keboola.component.exceptions import UserException


class ErrorMessageMapper:
    """Maps technical errors to user-friendly messages."""

    # Mapping of error patterns to user-friendly messages
    ERROR_PATTERNS: Dict[str, str] = {
        # Database connection errors
        r"connection.*refused": "Could not connect to the database. Please check your connection settings and ensure the database is running.",
        r"authentication.*failed": "Database authentication failed. Please check your username and password.",
        r"timeout": "The operation timed out. Please check your network connection or try again later.",

        # Data validation errors
        r"Unable to parse UUID": "Invalid record ID format. Each record must have a valid UUID format.",
        r"INVALID_ARGUMENT": "Invalid data format. Please check your input data format.",
        r"null value in column.*violates not-null constraint": "Missing required data. Please ensure all required fields are filled.",

        # Vector store specific errors
        r"collection.*not found": "Database collection not found. Please check if the collection exists and you have proper access.",
        r"index.*not found": "Vector index not found. Please verify the index name in your configuration.",
        r"dimension mismatch": "Embedding dimension mismatch. Please ensure your embeddings match the expected dimension.",

        # API errors
        r"rate limit": "API rate limit exceeded. Please wait a moment before trying again.",
        r"quota exceeded": "API quota exceeded. Please check your subscription limits.",
        r"api key.*invalid": "Invalid API key. Please check your API key configuration.",

        # General errors
        r"permission denied": "Access denied. Please check your permissions.",
        r"out of memory": "System memory limit reached. Try processing data in smaller batches.",
        r"disk space": "Insufficient disk space. Please free up some space and try again."
    }

    @classmethod
    def get_user_friendly_message(cls, error: Exception) -> str:
        """Convert technical error message to user-friendly message.
        Args:
            error: The original exception
        Returns:
            A user-friendly error message
        """
        error_msg = str(error).lower()

        # Try to match error patterns
        for pattern, message in cls.ERROR_PATTERNS.items():
            if re.search(pattern.lower(), error_msg):
                return message

        # If no specific pattern matches, return a generic message with the original error
        return f"An unexpected error occurred: {str(error)}"


def handle_exception(error: Exception, context: Optional[str] = None) -> UserException:
    """Convert any exception to UserException with a friendly message.
    Args:
        error: The original exception
        context: Optional context about where the error occurred
    Returns:
        UserException with user-friendly message
    """
    # If it's already a UserException, just return it
    if isinstance(error, UserException):
        return error

    # Get user-friendly message
    friendly_message = ErrorMessageMapper.get_user_friendly_message(error)

    # Add context if provided
    if context:
        friendly_message = f"{context}: {friendly_message}"

    return UserException(friendly_message)


def friendly_error_handler(func):
    """Decorator to convert function exceptions to user-friendly messages."""

    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            raise handle_exception(e, f"Error in {func.__name__}")

    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise handle_exception(e, f"Error in {func.__name__}")

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
