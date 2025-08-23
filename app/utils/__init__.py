"""
Utility functions and helpers.

This package contains common utility functions used throughout the application:
- String manipulation and text processing utilities
- Date and time handling functions
- File and path utilities
- Validation and sanitization functions
- Encoding and decoding utilities
- Performance and profiling helpers
- Common data structures and algorithms
"""

from .text_processing import (
    clean_text,
    normalize_text,
    remove_hallucinations,
    detect_language,
    split_sentences,
    merge_text_segments
)

from .time_utils import (
    format_duration,
    parse_timestamp,
    get_utc_now,
    convert_timezone,
    time_range_overlap,
    calculate_time_offset
)

from .file_utils import (
    get_file_extension,
    generate_unique_filename,
    create_directory_if_not_exists,
    calculate_file_hash,
    get_file_size,
    is_audio_file,
    sanitize_filename
)

from .validation import (
    validate_email,
    validate_phone_number,
    validate_url,
    validate_uuid,
    sanitize_input,
    validate_audio_format,
    validate_file_size
)

from .encoding import (
    encode_base64,
    decode_base64,
    hash_password,
    generate_random_string,
    create_jwt_token,
    decode_jwt_token
)

from .performance import (
    timer,
    memory_profiler,
    rate_limiter,
    cached_property,
    memoize,
    profile_function
)

from .audio_utils import (
    convert_audio_format,
    extract_audio_features,
    normalize_audio_levels,
    detect_silence,
    split_audio_segments,
    merge_audio_files
)

from .data_structures import (
    LRUCache,
    CircularBuffer,
    PriorityQueue,
    BloomFilter,
    TimedCache
)

__all__ = [
    # Text processing
    "clean_text",
    "normalize_text",
    "remove_hallucinations",
    "detect_language",
    "split_sentences",
    "merge_text_segments",
    
    # Time utilities
    "format_duration",
    "parse_timestamp", 
    "get_utc_now",
    "convert_timezone",
    "time_range_overlap",
    "calculate_time_offset",
    
    # File utilities
    "get_file_extension",
    "generate_unique_filename",
    "create_directory_if_not_exists",
    "calculate_file_hash",
    "get_file_size",
    "is_audio_file",
    "sanitize_filename",
    
    # Validation
    "validate_email",
    "validate_phone_number",
    "validate_url",
    "validate_uuid",
    "sanitize_input",
    "validate_audio_format",
    "validate_file_size",
    
    # Encoding
    "encode_base64",
    "decode_base64", 
    "hash_password",
    "generate_random_string",
    "create_jwt_token",
    "decode_jwt_token",
    
    # Performance
    "timer",
    "memory_profiler",
    "rate_limiter",
    "cached_property",
    "memoize",
    "profile_function",
    
    # Audio utilities
    "convert_audio_format",
    "extract_audio_features",
    "normalize_audio_levels",
    "detect_silence",
    "split_audio_segments", 
    "merge_audio_files",
    
    # Data structures
    "LRUCache",
    "CircularBuffer",
    "PriorityQueue",
    "BloomFilter",
    "TimedCache"
]