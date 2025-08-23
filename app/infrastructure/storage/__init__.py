"""
Storage infrastructure for file management.

This package provides file storage abstractions and implementations:
- Base file storage interface and abstract classes
- Local file system storage implementation
- Cloud storage implementations (S3, GCS, Azure Blob)
- Audio file specific management and processing
- File metadata extraction and validation
- Storage optimization and lifecycle management
"""

from .base import (
    FileStorage,
    FileManager,
    StorageError,
    FileNotFoundError,
    StoragePermissionError
)

from .file_storage import LocalFileStorage
from .audio_storage import (
    AudioFileManager,
    AudioProcessor,
    AudioMetadataExtractor
)

from .cloud_storage import (
    S3FileStorage,
    GCSFileStorage,
    AzureBlobStorage
)

from .utils import (
    generate_file_path,
    validate_file_extension,
    calculate_file_hash,
    compress_file,
    extract_file_metadata
)

# Dependency injection
from .dependencies import (
    get_file_manager,
    get_audio_file_manager,
    get_file_storage,
    get_audio_processor
)

__all__ = [
    # Base interfaces
    "FileStorage",
    "FileManager",
    "StorageError",
    "FileNotFoundError",
    "StoragePermissionError",
    
    # Local storage
    "LocalFileStorage",
    
    # Audio storage
    "AudioFileManager",
    "AudioProcessor",
    "AudioMetadataExtractor",
    
    # Cloud storage
    "S3FileStorage",
    "GCSFileStorage",
    "AzureBlobStorage",
    
    # Utilities
    "generate_file_path",
    "validate_file_extension",
    "calculate_file_hash",
    "compress_file",
    "extract_file_metadata",
    
    # Dependencies
    "get_file_manager",
    "get_audio_file_manager",
    "get_file_storage",
    "get_audio_processor"
]