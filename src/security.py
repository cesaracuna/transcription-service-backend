"""
Security utilities for file upload validation and sanitization.
Implements defensive security measures to prevent malicious file uploads.
"""

import os
import re
import uuid
import magic
import logging
import hashlib
from typing import Tuple, Optional, Set
from pathlib import Path
from werkzeug.utils import secure_filename
from .logging_config import get_security_logger

logger = get_security_logger(__name__)
general_logger = logging.getLogger(__name__)

# Security configuration
MAX_FILENAME_LENGTH = 255
MAX_FILE_SIZE_MB = 500  # 500MB limit for audio files
ALLOWED_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
ALLOWED_MIME_TYPES = {
    'audio/wav', 'audio/wave', 'audio/x-wav',
    'audio/mpeg', 'audio/mp3',
    'audio/flac', 'audio/x-flac',
    'audio/aac', 'audio/x-aac',
    'audio/ogg', 'audio/x-ogg',
    'audio/mp4', 'audio/x-m4a',
    'audio/x-ms-wma'
}

# Dangerous patterns in filenames
DANGEROUS_PATTERNS = [
    r'\.\./',           # Directory traversal
    r'\.\.\\',          # Windows directory traversal
    r'[<>:"|?*]',       # Windows invalid characters
    r'[\x00-\x1f]',     # Control characters
    r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)',  # Windows reserved names
    r'^\.',             # Hidden files (Unix)
    r'\.$',             # Files ending with dot
]


class SecurityError(Exception):
    """Exception raised for security validation failures."""
    pass


def sanitize_filename(filename: str, max_length: int = MAX_FILENAME_LENGTH) -> str:
    """
    Sanitizes a filename by removing dangerous characters.
    
    Args:
        filename: Original filename
        max_length: Maximum allowed length
        
    Returns:
        Sanitized filename
        
    Raises:
        SecurityError: If filename is invalid or dangerous
    """
    if not filename or not isinstance(filename, str):
        raise SecurityError("Invalid or empty filename")
    
    logger.debug(f"Sanitizing filename: '{filename}'")
    
    # Check dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, filename, re.IGNORECASE):
            logger.warning(f"Dangerous pattern detected in filename '{filename}': {pattern}")
            raise SecurityError(f"Filename contains disallowed characters")
    
    # Use werkzeug for basic sanitization
    safe_filename = secure_filename(filename)
    
    if not safe_filename:
        # If secure_filename returns empty, generate generic name
        safe_filename = "uploaded_file"
    
    # Truncate if too long
    if len(safe_filename) > max_length:
        name, ext = os.path.splitext(safe_filename)
        max_name_length = max_length - len(ext) - 10  # Space for UUID
        safe_filename = name[:max_name_length] + ext
    
    logger.debug(f"File sanitized: '{filename}' -> '{safe_filename}'")
    return safe_filename


def generate_secure_filename(original_filename: str, prefix: str = "") -> str:
    """
    Generates a secure filename with UUID and sanitized name.
    
    Args:
        original_filename: Original filename
        prefix: Optional prefix for the file
        
    Returns:
        Unique secure filename
    """
    # Sanitize the original name
    safe_name = sanitize_filename(original_filename)
    
    # Get the extension
    _, ext = os.path.splitext(safe_name)
    
    # Generate unique UUID
    unique_id = str(uuid.uuid4())
    
    # Build final name
    if prefix:
        secure_name = f"{prefix}_{unique_id}_{safe_name}"
    else:
        secure_name = f"{unique_id}_{safe_name}"
    
    logger.info(f"Filename generated: '{original_filename}' -> '{secure_name}'")
    return secure_name


def validate_file_extension(filename: str, allowed_extensions: Set[str] = None) -> bool:
    """
    Validates that the file extension is allowed.
    
    Args:
        filename: Filename to validate
        allowed_extensions: Allowed extensions (default: ALLOWED_AUDIO_EXTENSIONS)
        
    Returns:
        True if extension is allowed
    """
    if allowed_extensions is None:
        allowed_extensions = ALLOWED_AUDIO_EXTENSIONS
    
    ext = os.path.splitext(filename)[1].lower()
    is_valid = ext in allowed_extensions
    
    if not is_valid:
        logger.warning(f"Extension not allowed: '{ext}' in file '{filename}'")
    else:
        logger.debug(f"Valid extension: '{ext}' for file '{filename}'")
    
    return is_valid


def validate_mime_type(file_path: str, allowed_types: Set[str] = None) -> Tuple[bool, str]:
    """
    Validates the actual MIME type of the file using libmagic.
    
    Args:
        file_path: Ruta al archivo
        allowed_types: Allowed MIME types (default: ALLOWED_MIME_TYPES)
        
    Returns:
        Tuple (is_valid, detected_mime_type)
        
    Raises:
        SecurityError: If MIME type cannot be determined
    """
    if allowed_types is None:
        allowed_types = ALLOWED_MIME_TYPES
    
    try:
        # Use python-magic for precise detection
        detected_mime = magic.from_file(file_path, mime=True)
        is_valid = detected_mime in allowed_types
        
        if not is_valid:
            logger.warning(f"MIME type not allowed: '{detected_mime}' for file '{file_path}'")
            logger.debug(f"Allowed types: {sorted(allowed_types)}")
        else:
            logger.debug(f"Valid MIME type: '{detected_mime}' for file '{file_path}'")
        
        return is_valid, detected_mime
        
    except Exception as e:
        logger.error(f"Error detecting MIME type for '{file_path}': {e}")
        raise SecurityError(f"Could not validate file type: {e}")


def validate_file_size(file_path: str, max_size_mb: int = MAX_FILE_SIZE_MB) -> bool:
    """
    Validates that the file size is within allowed limits.
    
    Args:
        file_path: Path to the file
        max_size_mb: Maximum size in MB
        
    Returns:
        True if size is valid
    """
    try:
        file_size = os.path.getsize(file_path)
        max_size_bytes = max_size_mb * 1024 * 1024
        
        is_valid = file_size <= max_size_bytes
        file_size_mb = file_size / (1024 * 1024)
        
        if not is_valid:
            logger.warning(f"File exceeds maximum size: {file_size_mb:.2f}MB > {max_size_mb}MB")
        else:
            logger.debug(f"Valid file size: {file_size_mb:.2f}MB")
        
        return is_valid
        
    except OSError as e:
        logger.error(f"Error checking file size '{file_path}': {e}")
        return False


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Calculates file hash for integrity and duplicate detection.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (sha256, md5, etc.)
        
    Returns:
        Hexadecimal hash of the file
    """
    hash_obj = hashlib.new(algorithm)
    
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        file_hash = hash_obj.hexdigest()
        logger.debug(f"Hash {algorithm} calculated for '{file_path}': {file_hash}")
        return file_hash
        
    except Exception as e:
        logger.error(f"Error calculating hash for '{file_path}': {e}")
        raise SecurityError(f"Error calculating file hash: {e}")


def comprehensive_file_validation(file_path: str, original_filename: str) -> dict:
    """
    Performs comprehensive security validation for a file.
    
    Args:
        file_path: Path to file on disk
        original_filename: Original filename
        
    Returns:
        Dictionary with validation results
        
    Raises:
        SecurityError: If any critical validation fails
    """
    logger.info(f"Starting comprehensive security validation for: '{original_filename}'")
    
    validation_results = {
        'filename': original_filename,
        'file_path': file_path,
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'detected_mime_type': None,
        'file_size_mb': 0,
        'file_hash': None
    }
    
    try:
        # 1. Validate extension
        if not validate_file_extension(original_filename):
            error_msg = f"File extension not allowed: {os.path.splitext(original_filename)[1]}"
            validation_results['errors'].append(error_msg)
            validation_results['is_valid'] = False
        
        # 2. Validate size
        if not validate_file_size(file_path):
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            validation_results['file_size_mb'] = file_size_mb
            error_msg = f"File exceeds maximum allowed size: {file_size_mb:.2f}MB"
            validation_results['errors'].append(error_msg)
            validation_results['is_valid'] = False
        else:
            validation_results['file_size_mb'] = os.path.getsize(file_path) / (1024 * 1024)
        
        # 3. Validar tipo MIME
        is_mime_valid, detected_mime = validate_mime_type(file_path)
        validation_results['detected_mime_type'] = detected_mime
        
        if not is_mime_valid:
            error_msg = f"File type not allowed: {detected_mime}"
            validation_results['errors'].append(error_msg)
            validation_results['is_valid'] = False
        
        # 4. Calculate hash for integrity
        try:
            validation_results['file_hash'] = calculate_file_hash(file_path)
        except SecurityError as e:
            validation_results['warnings'].append(f"Could not calculate hash: {e}")
        
        # 5. Additional security checks
        _perform_additional_security_checks(file_path, validation_results)
        
        if validation_results['is_valid']:
            logger.info(f"Security validation completed successfully for: '{original_filename}'")
        else:
            logger.error(f"Security validation failed for: '{original_filename}'. Errors: {validation_results['errors']}")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Error during security validation: {e}", exc_info=True)
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Internal validation error: {e}")
        return validation_results


def _perform_additional_security_checks(file_path: str, validation_results: dict) -> None:
    """
    Performs additional security checks on the file.
    
    Args:
        file_path: Path to the file
        validation_results: Results dictionary to update
    """
    try:
        # Verify it's not a symbolic link
        if os.path.islink(file_path):
            validation_results['warnings'].append("File is a symbolic link")
        
        # Verify file permissions
        if not os.access(file_path, os.R_OK):
            validation_results['errors'].append("File is not readable")
            validation_results['is_valid'] = False
        
        # Verify the file is not empty
        if os.path.getsize(file_path) == 0:
            validation_results['errors'].append("File is empty")
            validation_results['is_valid'] = False
        
        # Verify basic audio file structure
        _basic_audio_structure_check(file_path, validation_results)
        
    except Exception as e:
        logger.warning(f"Error in additional security checks: {e}")
        validation_results['warnings'].append(f"Error in additional checks: {e}")


def _basic_audio_structure_check(file_path: str, validation_results: dict) -> None:
    """
    Basic audio file structure verification.
    
    Args:
        file_path: Path to the file
        validation_results: Results dictionary to update
    """
    try:
        # Read first bytes to verify known headers
        with open(file_path, 'rb') as f:
            header = f.read(12)
        
        # Verify headers of common audio formats
        audio_signatures = {
            b'RIFF': 'WAV',
            b'ID3': 'MP3', 
            b'\xff\xfb': 'MP3',
            b'\xff\xf3': 'MP3',
            b'\xff\xf2': 'MP3',
            b'fLaC': 'FLAC',
            b'OggS': 'OGG'
        }
        
        header_found = False
        for signature, format_name in audio_signatures.items():
            if header.startswith(signature):
                logger.debug(f"Valid audio header detected: {format_name}")
                header_found = True
                break
        
        if not header_found:
            validation_results['warnings'].append("No recognized audio header detected")
            
    except Exception as e:
        logger.debug(f"Could not verify audio structure: {e}")


def create_secure_storage_path(base_path: str, filename: str) -> str:
    """
    Creates a secure storage path avoiding directory traversal.
    
    Args:
        base_path: Safe base directory
        filename: Filename
        
    Returns:
        Complete secure path
    """
    # Resolve absolute paths to avoid directory traversal
    safe_base = os.path.abspath(base_path)
    safe_filename = os.path.basename(filename)  # Only the name, no paths
    
    full_path = os.path.join(safe_base, safe_filename)
    
    # Verify the resulting path is within the base directory
    if not os.path.abspath(full_path).startswith(safe_base):
        raise SecurityError("Directory traversal attempt detected")
    
    return full_path