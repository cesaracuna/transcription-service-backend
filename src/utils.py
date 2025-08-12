"""
Utility functions for audio and text processing.
Enhanced with comprehensive logging, error handling, and performance monitoring.
"""

import time
import zlib
import numpy as np
import logging
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import Counter, defaultdict

from .constants import SILENCE_RMS_THRESHOLD

# Create module-specific logger
logger = logging.getLogger(__name__)


@dataclass
class TimestampMetrics:
    """Data class to track timestamp processing metrics."""
    total_processed: int = 0
    successful_conversions: int = 0
    failed_conversions: int = 0
    processing_time: float = 0.0
    invalid_formats: List[str] = None
    
    def __post_init__(self):
        if self.invalid_formats is None:
            self.invalid_formats = []


@dataclass
class SpeakerRemappingMetrics:
    """Data class to track speaker remapping metrics."""
    original_speakers: int = 0
    remapped_speakers: int = 0
    segments_processed: int = 0
    words_processed: int = 0
    processing_time: float = 0.0
    speaker_mapping: Dict[str, str] = None
    
    def __post_init__(self):
        if self.speaker_mapping is None:
            self.speaker_mapping = {}


class UtilityError(Exception):
    """Custom exception for utility function errors."""
    pass


def validate_timestamp_format(ts_str: str) -> bool:
    """
    Validates if a timestamp string matches the expected HH:MM:SS.mmm format.
    
    Args:
        ts_str: Timestamp string to validate
        
    Returns:
        True if format is valid, False otherwise
    """
    if not isinstance(ts_str, str):
        return False
    
    # Pattern for HH:MM:SS.mmm format
    pattern = r'^\d{1,2}:\d{2}:\d{2}\.\d{3}$'
    return bool(re.match(pattern, ts_str))


def format_timestamp(seconds: float) -> str:
    """
    Converts seconds to timestamp format HH:MM:SS.mmm with enhanced validation and logging.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp as string
        
    Raises:
        UtilityError: If input is invalid
    """
    logger.debug(f"Formatting timestamp: {seconds} seconds")
    
    try:
        # Validate input
        if not isinstance(seconds, (int, float)):
            raise UtilityError(f"Invalid input type for seconds: {type(seconds)}. Expected int or float.")
        
        if seconds < 0:
            logger.warning(f"Negative timestamp value: {seconds}s. Using absolute value.")
            seconds = abs(seconds)
        
        if seconds > 86400:  # More than 24 hours
            logger.warning(f"Very large timestamp value: {seconds}s ({seconds/3600:.1f} hours)")
        
        # Calculate components
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds * 1000) % 1000)
        
        # Format timestamp
        formatted = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
        
        logger.debug(f"Formatted {seconds}s -> {formatted}")
        
        # Validate the result
        if not validate_timestamp_format(formatted):
            raise UtilityError(f"Generated invalid timestamp format: {formatted}")
        
        return formatted
        
    except Exception as e:
        logger.error(f"Error formatting timestamp {seconds}: {e}", exc_info=True)
        raise UtilityError(f"Timestamp formatting failed: {e}")


def parse_timestamp_to_seconds(ts_str: str) -> float:
    """
    Converts an H:M:S.ms timestamp to total seconds with enhanced validation and logging.
    
    Args:
        ts_str: Timestamp in string format
        
    Returns:
        Total time in seconds
        
    Raises:
        UtilityError: If parsing fails
    """
    logger.debug(f"Parsing timestamp: '{ts_str}'")
    
    try:
        # Validate input
        if not isinstance(ts_str, str):
            raise UtilityError(f"Invalid input type: {type(ts_str)}. Expected string.")
        
        if not ts_str or not ts_str.strip():
            raise UtilityError("Empty or whitespace-only timestamp string")
        
        ts_str = ts_str.strip()
        
        # Pre-validate format
        if not validate_timestamp_format(ts_str):
            logger.warning(f"Timestamp format may be invalid: '{ts_str}'. Attempting to parse anyway.")
        
        # Parse components
        try:
            h, m, s_ms = ts_str.split(':')
            s, ms = s_ms.split('.')
        except ValueError as e:
            raise UtilityError(f"Invalid timestamp format '{ts_str}'. Expected HH:MM:SS.mmm format. Error: {e}")
        
        # Convert to integers with validation
        try:
            h_int = int(h)
            m_int = int(m)
            s_int = int(s)
            ms_int = int(ms)
        except ValueError as e:
            raise UtilityError(f"Non-numeric components in timestamp '{ts_str}': {e}")
        
        # Validate ranges
        if not (0 <= h_int <= 23):
            logger.warning(f"Hour value out of normal range: {h_int} (0-23 expected)")
        if not (0 <= m_int <= 59):
            raise UtilityError(f"Invalid minute value: {m_int} (must be 0-59)")
        if not (0 <= s_int <= 59):
            raise UtilityError(f"Invalid second value: {s_int} (must be 0-59)")
        if not (0 <= ms_int <= 999):
            raise UtilityError(f"Invalid millisecond value: {ms_int} (must be 0-999)")
        
        # Calculate total seconds
        total_seconds = h_int * 3600 + m_int * 60 + s_int + ms_int / 1000.0
        
        logger.debug(f"Parsed '{ts_str}' -> {total_seconds}s")
        
        return total_seconds
        
    except UtilityError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error parsing timestamp '{ts_str}': {e}", exc_info=True)
        raise UtilityError(f"Timestamp parsing failed: {e}")


def batch_parse_timestamps(timestamps: List[str]) -> Tuple[List[float], TimestampMetrics]:
    """
    Parses multiple timestamps with comprehensive metrics and error handling.
    
    Args:
        timestamps: List of timestamp strings to parse
        
    Returns:
        Tuple of (parsed_seconds_list, metrics)
    """
    start_time = time.time()
    metrics = TimestampMetrics()
    results = []
    
    logger.info(f"Starting batch timestamp parsing for {len(timestamps)} timestamps")
    
    for i, ts_str in enumerate(timestamps):
        metrics.total_processed += 1
        
        try:
            seconds = parse_timestamp_to_seconds(ts_str)
            results.append(seconds)
            metrics.successful_conversions += 1
            
        except UtilityError as e:
            logger.warning(f"Failed to parse timestamp {i}: '{ts_str}' - {e}")
            metrics.failed_conversions += 1
            metrics.invalid_formats.append(ts_str)
            results.append(0.0)  # Default fallback value
    
    metrics.processing_time = time.time() - start_time
    
    logger.info(f"Batch timestamp parsing completed in {metrics.processing_time:.3f}s:")
    logger.info(f"  - Total processed: {metrics.total_processed}")
    logger.info(f"  - Successful: {metrics.successful_conversions}")
    logger.info(f"  - Failed: {metrics.failed_conversions}")
    
    if metrics.failed_conversions > 0:
        logger.warning(f"Failed timestamp formats: {metrics.invalid_formats[:5]}" + 
                      (f" and {len(metrics.invalid_formats)-5} more" if len(metrics.invalid_formats) > 5 else ""))
    
    return results, metrics


def is_silent(audio_chunk: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> Tuple[float, bool]:
    """
    Calculates RMS energy and determines if the chunk is silent with enhanced validation.
    
    Args:
        audio_chunk: Audio chunk as numpy array
        threshold: Energy threshold to consider silence
        
    Returns:
        Tuple with (rms_value, is_silent)
        
    Raises:
        UtilityError: If audio chunk is invalid
    """
    logger.debug(f"Analyzing audio chunk silence (length: {len(audio_chunk) if audio_chunk is not None else 'None'}, threshold: {threshold})")
    
    try:
        # Validate input
        if audio_chunk is None:
            raise UtilityError("Audio chunk is None")
        
        if not isinstance(audio_chunk, np.ndarray):
            logger.debug(f"Converting audio chunk from {type(audio_chunk)} to numpy array")
            audio_chunk = np.array(audio_chunk)
        
        if audio_chunk.size == 0:
            logger.warning("Empty audio chunk provided")
            return 0.0, True
        
        if len(audio_chunk.shape) > 1:
            logger.warning(f"Multi-dimensional audio chunk with shape {audio_chunk.shape}. Flattening.")
            audio_chunk = audio_chunk.flatten()
        
        # Validate threshold
        if not isinstance(threshold, (int, float)) or threshold < 0:
            logger.warning(f"Invalid threshold value: {threshold}. Using default {SILENCE_RMS_THRESHOLD}")
            threshold = SILENCE_RMS_THRESHOLD
        
        # Calculate RMS with enhanced precision
        audio_float32 = np.array(audio_chunk, dtype=np.float32)
        
        # Check for unusual values
        if np.any(np.isnan(audio_float32)):
            logger.warning("Audio chunk contains NaN values")
            audio_float32 = np.nan_to_num(audio_float32)
        
        if np.any(np.isinf(audio_float32)):
            logger.warning("Audio chunk contains infinite values")
            audio_float32 = np.nan_to_num(audio_float32)
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio_float32 ** 2))
        is_silent_flag = rms < threshold
        
        # Log analysis results
        logger.debug(f"Silence analysis: RMS={rms:.6f}, threshold={threshold:.6f}, silent={is_silent_flag}")
        
        # Additional statistics for debugging
        if logger.isEnabledFor(logging.DEBUG):
            peak = np.max(np.abs(audio_float32))
            logger.debug(f"Audio stats: peak={peak:.6f}, mean={np.mean(audio_float32):.6f}, std={np.std(audio_float32):.6f}")
        
        return float(rms), is_silent_flag
        
    except Exception as e:
        logger.error(f"Error in silence detection: {e}", exc_info=True)
        raise UtilityError(f"Silence detection failed: {e}")


def get_compression_ratio(text: str) -> float:
    """
    Calculates the compression ratio of a text using zlib with enhanced validation and logging.
    
    Args:
        text: Text to analyze
        
    Returns:
        Compression ratio (lower value indicates more repetition)
        
    Raises:
        UtilityError: If compression analysis fails
    """
    logger.debug(f"Analyzing text compression (length: {len(text) if text else 'None'})")
    
    try:
        # Validate input
        if text is None:
            logger.debug("Text is None, returning ratio 1.0")
            return 1.0
        
        if not isinstance(text, str):
            logger.warning(f"Converting text from {type(text)} to string")
            text = str(text)
        
        # Handle empty or whitespace-only text
        stripped_text = text.strip()
        if not stripped_text:
            logger.debug("Empty or whitespace-only text, returning ratio 1.0")
            return 1.0
        
        # Encode text
        try:
            text_bytes = stripped_text.encode("utf-8")
        except UnicodeEncodeError as e:
            logger.error(f"Unicode encoding error: {e}")
            raise UtilityError(f"Text encoding failed: {e}")
        
        original_size = len(text_bytes)
        
        # Compress text
        try:
            compressed_bytes = zlib.compress(text_bytes)
            compressed_size = len(compressed_bytes)
        except zlib.error as e:
            logger.error(f"Compression error: {e}")
            raise UtilityError(f"Text compression failed: {e}")
        
        # Calculate ratio
        if original_size == 0:
            ratio = 1.0
        else:
            ratio = compressed_size / original_size
        
        # Log analysis results
        logger.debug(f"Compression analysis: original={original_size}B, compressed={compressed_size}B, ratio={ratio:.4f}")
        
        # Additional analysis for debugging
        if logger.isEnabledFor(logging.DEBUG):
            char_counts = Counter(stripped_text)
            unique_chars = len(char_counts)
            most_common = char_counts.most_common(3)
            logger.debug(f"Text stats: unique_chars={unique_chars}, most_common={most_common}")
            
            # Detect highly repetitive patterns
            if ratio < 0.3:
                logger.debug(f"Highly repetitive text detected (ratio={ratio:.4f})")
            elif ratio > 0.9:
                logger.debug(f"Low compression text detected (ratio={ratio:.4f})")
        
        return ratio
        
    except UtilityError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in compression analysis: {e}", exc_info=True)
        raise UtilityError(f"Compression analysis failed: {e}")


def analyze_text_patterns(texts: List[str]) -> Dict[str, Any]:
    """
    Analyzes patterns in a collection of texts for insights and reporting.
    
    Args:
        texts: List of texts to analyze
        
    Returns:
        Dictionary containing pattern analysis results
    """
    if not texts:
        logger.debug("No texts to analyze")
        return {"total_texts": 0}
    
    logger.debug(f"Analyzing patterns in {len(texts)} texts")
    
    # Calculate compression ratios
    compression_ratios = []
    repetitive_texts = []
    
    for i, text in enumerate(texts):
        try:
            ratio = get_compression_ratio(text)
            compression_ratios.append(ratio)
            
            if ratio < 0.4:  # Threshold for repetitive content
                repetitive_texts.append({"index": i, "text": text[:100], "ratio": ratio})
                
        except UtilityError:
            compression_ratios.append(1.0)  # Default for failed analysis
    
    # Calculate statistics
    if compression_ratios:
        avg_ratio = sum(compression_ratios) / len(compression_ratios)
        min_ratio = min(compression_ratios)
        max_ratio = max(compression_ratios)
    else:
        avg_ratio = min_ratio = max_ratio = 1.0
    
    # Text length statistics
    text_lengths = [len(text) for text in texts if text]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    
    analysis_results = {
        "total_texts": len(texts),
        "average_compression_ratio": avg_ratio,
        "compression_range": (min_ratio, max_ratio),
        "repetitive_texts_count": len(repetitive_texts),
        "average_text_length": avg_length,
        "repetitive_examples": repetitive_texts[:3]  # First 3 examples
    }
    
    logger.debug(f"Text pattern analysis: {analysis_results}")
    return analysis_results


def remap_speakers_chronologically(segments_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remaps speaker labels to chronological order with comprehensive logging and metrics.
    
    Args:
        segments_data: List of segments with speaker information
        
    Returns:
        List of segments with remapped speakers
        
    Raises:
        UtilityError: If remapping fails
    """
    start_time = time.time()
    metrics = SpeakerRemappingMetrics()
    
    logger.info(f"Starting chronological speaker remapping for {len(segments_data)} segments")
    
    try:
        # Validate input
        if not segments_data:
            logger.warning("Empty segments list provided for speaker remapping")
            return segments_data
        
        if not isinstance(segments_data, list):
            raise UtilityError(f"Invalid input type: {type(segments_data)}. Expected list.")
        
        # Build speaker mapping
        speaker_map = {}
        new_speaker_index = 1
        first_appearances = {}  # Track first appearance time for each speaker
        
        logger.debug("Building speaker mapping based on chronological appearance...")
        
        # First pass: find first appearance of each speaker and build the map
        for i, segment in enumerate(segments_data):
            if not isinstance(segment, dict):
                logger.warning(f"Segment {i} is not a dictionary. Skipping.")
                continue
            
            original_speaker = segment.get('speaker')
            if not original_speaker or original_speaker == "UNKNOWN":
                continue
            
            if original_speaker not in speaker_map:
                # Use :02d format to get leading zeros (01, 02, etc.)
                new_label = f"SPEAKER_{new_speaker_index:02d}"
                speaker_map[original_speaker] = new_label
                first_appearances[original_speaker] = i  # Track first appearance
                
                logger.debug(f"Mapping {original_speaker} -> {new_label} (first seen at segment {i})")
                new_speaker_index += 1
        
        metrics.original_speakers = len(speaker_map)
        
        if not speaker_map:
            logger.info("No speakers found to remap")
            return segments_data
        
        logger.info(f"Speaker mapping created with {len(speaker_map)} speakers:")
        for orig, new in speaker_map.items():
            first_idx = first_appearances.get(orig, -1)
            logger.info(f"  - {orig} -> {new} (first appearance: segment {first_idx})")
        
        metrics.speaker_mapping = speaker_map.copy()
        
        # Second pass: apply the new labels to all segments and words
        logger.debug("Applying speaker remapping to segments and words...")
        
        for i, segment in enumerate(segments_data):
            if not isinstance(segment, dict):
                continue
            
            metrics.segments_processed += 1
            
            # Remap segment speaker
            original_segment_speaker = segment.get('speaker')
            if original_segment_speaker in speaker_map:
                old_speaker = segment['speaker']
                segment['speaker'] = speaker_map[original_segment_speaker]
                logger.debug(f"Segment {i}: {old_speaker} -> {segment['speaker']}")
            
            # Remap word-level speakers if present
            words = segment.get("words", [])
            for word in words:
                if isinstance(word, dict):
                    metrics.words_processed += 1
                    original_word_speaker = word.get('speaker')
                    if original_word_speaker in speaker_map:
                        word['speaker'] = speaker_map[original_word_speaker]
        
        metrics.remapped_speakers = len(speaker_map)
        metrics.processing_time = time.time() - start_time
        
        # Log final statistics
        logger.info("Speaker remapping completed successfully:")
        logger.info(f"  - Processing time: {metrics.processing_time:.3f}s")
        logger.info(f"  - Original speakers: {metrics.original_speakers}")
        logger.info(f"  - Remapped speakers: {metrics.remapped_speakers}")
        logger.info(f"  - Segments processed: {metrics.segments_processed}")
        logger.info(f"  - Words processed: {metrics.words_processed}")
        
        # Validate result
        _validate_remapping_result(segments_data, speaker_map)
        
        return segments_data
        
    except UtilityError:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Speaker remapping failed after {processing_time:.3f}s: {e}", exc_info=True)
        raise UtilityError(f"Speaker remapping failed: {e}")


def _validate_remapping_result(segments_data: List[Dict[str, Any]], speaker_map: Dict[str, str]) -> None:
    """
    Validates the result of speaker remapping for consistency.
    
    Args:
        segments_data: Remapped segments data
        speaker_map: Original to new speaker mapping
    """
    logger.debug("Validating speaker remapping result...")
    
    found_speakers = set()
    expected_speakers = set(speaker_map.values())
    
    for segment in segments_data:
        if isinstance(segment, dict):
            speaker = segment.get('speaker')
            if speaker and speaker != "UNKNOWN":
                found_speakers.add(speaker)
    
    # Check if all expected speakers are found
    missing_speakers = expected_speakers - found_speakers
    unexpected_speakers = found_speakers - expected_speakers - {"UNKNOWN"}
    
    if missing_speakers:
        logger.warning(f"Expected speakers not found in result: {missing_speakers}")
    
    if unexpected_speakers:
        logger.warning(f"Unexpected speakers found in result: {unexpected_speakers}")
    
    logger.debug(f"Validation complete. Found speakers: {sorted(found_speakers)}")


def calculate_audio_statistics(audio_chunks: List[np.ndarray]) -> Dict[str, Any]:
    """
    Calculates comprehensive statistics for a collection of audio chunks.
    
    Args:
        audio_chunks: List of audio chunks as numpy arrays
        
    Returns:
        Dictionary containing audio statistics
    """
    if not audio_chunks:
        logger.debug("No audio chunks to analyze")
        return {"total_chunks": 0}
    
    logger.debug(f"Calculating statistics for {len(audio_chunks)} audio chunks")
    
    stats = {
        "total_chunks": len(audio_chunks),
        "total_samples": 0,
        "silent_chunks": 0,
        "rms_values": [],
        "chunk_lengths": [],
        "processing_time": 0.0
    }
    
    start_time = time.time()
    
    for i, chunk in enumerate(audio_chunks):
        try:
            if chunk is not None and len(chunk) > 0:
                stats["total_samples"] += len(chunk)
                stats["chunk_lengths"].append(len(chunk))
                
                # Calculate RMS for silence detection
                rms, is_silent_flag = is_silent(chunk)
                stats["rms_values"].append(rms)
                
                if is_silent_flag:
                    stats["silent_chunks"] += 1
            else:
                logger.debug(f"Empty or None chunk at index {i}")
                
        except Exception as e:
            logger.warning(f"Error processing audio chunk {i}: {e}")
    
    stats["processing_time"] = time.time() - start_time
    
    # Calculate derived statistics
    if stats["rms_values"]:
        stats["average_rms"] = sum(stats["rms_values"]) / len(stats["rms_values"])
        stats["max_rms"] = max(stats["rms_values"])
        stats["min_rms"] = min(stats["rms_values"])
    
    if stats["chunk_lengths"]:
        stats["average_chunk_length"] = sum(stats["chunk_lengths"]) / len(stats["chunk_lengths"])
        stats["max_chunk_length"] = max(stats["chunk_lengths"])
        stats["min_chunk_length"] = min(stats["chunk_lengths"])
    
    stats["silence_ratio"] = stats["silent_chunks"] / stats["total_chunks"] if stats["total_chunks"] > 0 else 0
    
    logger.debug(f"Audio statistics calculated: {stats}")
    return stats