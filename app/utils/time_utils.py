"""
Time utility functions.
"""

from datetime import datetime, timezone
from typing import Optional


def get_utc_now() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    else:
        return f"{minutes:02d}:{seconds:06.3f}"


def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse timestamp string to datetime."""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        return None


def convert_timezone(dt: datetime, target_tz: timezone) -> datetime:
    """Convert datetime to target timezone."""
    return dt.astimezone(target_tz)


def time_range_overlap(start1: float, end1: float, start2: float, end2: float) -> bool:
    """Check if two time ranges overlap."""
    return start1 < end2 and start2 < end1


def calculate_time_offset(timestamp1: datetime, timestamp2: datetime) -> float:
    """Calculate time offset between two timestamps in seconds."""
    return (timestamp2 - timestamp1).total_seconds()