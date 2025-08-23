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

from .time_utils import (
    format_duration,
    parse_timestamp,
    get_utc_now,
    convert_timezone,
    time_range_overlap,
    calculate_time_offset
)

# Import other modules as they are implemented
try:
    from .text_processing import *
except ImportError:
    pass
    
try:
    from .file_utils import *
except ImportError:
    pass
    
try:
    from .validation import *
except ImportError:
    pass
    
try:
    from .encoding import *
except ImportError:
    pass
    
try:
    from .performance import *
except ImportError:
    pass
    
try:
    from .audio_utils import *
except ImportError:
    pass
    
try:
    from .data_structures import *
except ImportError:
    pass

__all__ = [
    # Time utilities (implemented)
    "format_duration",
    "parse_timestamp", 
    "get_utc_now",
    "convert_timezone",
    "time_range_overlap",
    "calculate_time_offset"
]