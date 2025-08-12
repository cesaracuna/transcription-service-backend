"""
Functions for detecting and handling hallucinations in transcriptions.
Enhanced with comprehensive logging, error handling, and performance monitoring.
"""

import time
import logging
from typing import List, Dict, Any, Set, Optional, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from .constants import SANDWICH_MAX_DURATION, SANDWICH_MAX_PAUSE
from .utils import parse_timestamp_to_seconds
from .models import Segment, Hallucination

# Create module-specific logger
logger = logging.getLogger(__name__)


@dataclass
class HallucinationDetectionMetrics:
    """Data class to track hallucination detection metrics."""
    segments_analyzed: int = 0
    sandwiched_segments_found: int = 0
    blocklist_matches: int = 0
    language_outliers: int = 0
    segments_marked_for_deletion: int = 0
    analysis_time: float = 0.0
    database_query_time: float = 0.0
    language_analysis_time: float = 0.0


class HallucinationDetectionError(Exception):
    """Custom exception for hallucination detection errors."""
    pass


@dataclass
class SegmentAnalysis:
    """Data class to hold detailed analysis of a segment."""
    segment: Segment
    is_sandwiched: bool
    is_short_interruption: bool
    are_gaps_small: bool
    matches_blocklist: bool
    is_language_outlier: bool
    duration: float
    prev_gap: float
    next_gap: float
    reason: str = ""


def validate_segments_input(segments: List[Segment]) -> None:
    """
    Validates the input segments list for hallucination detection.
    
    Args:
        segments: List of segments to validate
        
    Raises:
        HallucinationDetectionError: If segments are invalid
    """
    if not segments:
        logger.warning("Empty segments list provided for hallucination detection")
        return
    
    if len(segments) < 3:
        logger.info(f"Only {len(segments)} segments provided - need at least 3 for sandwich detection")
        return
    
    logger.debug(f"Validating {len(segments)} segments for hallucination detection")
    
    # Check for required attributes
    required_attrs = ['speaker', 'start_timestamp', 'end_timestamp', 'text', 'language']
    for i, segment in enumerate(segments):
        for attr in required_attrs:
            if not hasattr(segment, attr):
                raise HallucinationDetectionError(f"Segment {i} missing required attribute: {attr}")
            
            value = getattr(segment, attr)
            if value is None:
                logger.warning(f"Segment {i} has None value for attribute {attr}")
    
    logger.debug("Segment validation completed successfully")


def analyze_segment_context(
    prev_segment: Segment, 
    current_segment: Segment, 
    next_segment: Segment,
    hallucination_texts: Set[str],
    primary_languages: Set[str]
) -> SegmentAnalysis:
    """
    Performs detailed analysis of a segment in its context.
    
    Args:
        prev_segment: Previous segment
        current_segment: Current segment to analyze
        next_segment: Next segment
        hallucination_texts: Set of blacklisted texts
        primary_languages: Set of primary languages
        
    Returns:
        Detailed analysis of the segment
    """
    logger.debug(f"Analyzing segment context for speaker {current_segment.speaker} "
                f"at {current_segment.start_timestamp}-{current_segment.end_timestamp}")
    
    analysis = SegmentAnalysis(segment=current_segment, is_sandwiched=False, 
                              is_short_interruption=False, are_gaps_small=False,
                              matches_blocklist=False, is_language_outlier=False,
                              duration=0.0, prev_gap=0.0, next_gap=0.0)
    
    try:
        # 1. Check if segment is "sandwiched"
        analysis.is_sandwiched = (
            prev_segment.speaker == next_segment.speaker and 
            current_segment.speaker != prev_segment.speaker
        )
        
        if not analysis.is_sandwiched:
            logger.debug(f"    -> Not sandwiched: prev={prev_segment.speaker}, "
                        f"current={current_segment.speaker}, next={next_segment.speaker}")
            return analysis

        logger.debug(f"    -> Sandwiched segment detected between {prev_segment.speaker} turns")

        # 2. Calculate timing information
        current_start = parse_timestamp_to_seconds(current_segment.start_timestamp)
        current_end = parse_timestamp_to_seconds(current_segment.end_timestamp)
        prev_end = parse_timestamp_to_seconds(prev_segment.end_timestamp)
        next_start = parse_timestamp_to_seconds(next_segment.start_timestamp)
        
        analysis.duration = current_end - current_start
        analysis.prev_gap = current_start - prev_end
        analysis.next_gap = next_start - current_end
        
        logger.debug(f"    -> Timing: duration={analysis.duration:.2f}s, "
                    f"prev_gap={analysis.prev_gap:.2f}s, next_gap={analysis.next_gap:.2f}s")

        # 3. Check if it's a short interruption
        analysis.is_short_interruption = analysis.duration < SANDWICH_MAX_DURATION
        analysis.are_gaps_small = (
            analysis.prev_gap < SANDWICH_MAX_PAUSE and 
            analysis.next_gap < SANDWICH_MAX_PAUSE
        )
        
        if not (analysis.is_short_interruption and analysis.are_gaps_small):
            logger.debug(f"    -> Not qualifying: short={analysis.is_short_interruption}, "
                        f"small_gaps={analysis.are_gaps_small}")
            return analysis

        logger.debug("    -> Qualifies as short interruption with small gaps")

        # 4. Check content for suspicious patterns
        text_to_check = current_segment.text.strip().lower() if current_segment.text else ""
        
        # Blocklist check
        analysis.matches_blocklist = text_to_check in hallucination_texts
        if analysis.matches_blocklist:
            logger.debug(f"    -> Matches blocklist: '{current_segment.text}'")
        
        # Language outlier check
        analysis.is_language_outlier = (
            current_segment.language not in primary_languages and 
            current_segment.language != 'unknown'
        )
        if analysis.is_language_outlier:
            logger.debug(f"    -> Language outlier: '{current_segment.language}' "
                        f"not in {primary_languages}")

        # 5. Build reason string
        reasons = []
        if analysis.matches_blocklist:
            reasons.append(f"Text '{current_segment.text}' is on the blocklist")
        if analysis.is_language_outlier:
            reasons.append(f"Language '{current_segment.language}' is atypical")
        
        analysis.reason = ". ".join(reasons)
        
        return analysis

    except Exception as e:
        logger.error(f"Error analyzing segment context: {e}", exc_info=True)
        raise HallucinationDetectionError(f"Segment analysis failed: {e}")


def detect_sandwiched_hallucinations(
    segments: List[Segment], 
    hallucination_texts: Set[str], 
    primary_languages: Set[str]
) -> List[Segment]:
    """
    Detects hallucinations using contextual "sandwich" logic with comprehensive logging.
    
    A "sandwich" hallucination is a short segment between two turns from the same speaker
    that contains suspicious content (blacklisted text or atypical language).
    
    Args:
        segments: List of segments ordered by timestamp
        hallucination_texts: Set of blacklisted texts
        primary_languages: Set of primary languages in the audio
        
    Returns:
        List of segments identified as hallucinations
        
    Raises:
        HallucinationDetectionError: If detection fails
    """
    start_time = time.time()
    metrics = HallucinationDetectionMetrics()
    
    logger.info(f"Starting sandwiched hallucination detection on {len(segments)} segments")
    logger.info(f"Detection parameters: max_duration={SANDWICH_MAX_DURATION}s, "
               f"max_pause={SANDWICH_MAX_PAUSE}s")
    logger.info(f"Blocklist size: {len(hallucination_texts)} patterns")
    logger.info(f"Primary languages: {primary_languages}")
    
    try:
        # Validate input
        validate_segments_input(segments)
        
        segments_to_delete = []
        detailed_analyses = []
        
        if len(segments) < 3:
            logger.info("Insufficient segments for sandwich detection - returning empty result")
            return segments_to_delete

        # Iterate from second to second-to-last to check context
        logger.debug("Starting segment-by-segment analysis...")
        
        for i in range(1, len(segments) - 1):
            metrics.segments_analyzed += 1
            
            prev_segment = segments[i - 1]
            current_segment = segments[i]
            next_segment = segments[i + 1]
            
            # Perform detailed analysis
            analysis = analyze_segment_context(
                prev_segment, current_segment, next_segment,
                hallucination_texts, primary_languages
            )
            
            detailed_analyses.append(analysis)
            
            # Track metrics
            if analysis.is_sandwiched:
                metrics.sandwiched_segments_found += 1
            if analysis.matches_blocklist:
                metrics.blocklist_matches += 1
            if analysis.is_language_outlier:
                metrics.language_outliers += 1
            
            # Determine if segment should be deleted
            if analysis.matches_blocklist or analysis.is_language_outlier:
                logger.warning(f"HALLUCINATION DETECTED - Segment {i}: {analysis.reason}")
                logger.warning(f"  -> Speaker: {current_segment.speaker}")
                logger.warning(f"  -> Time: {current_segment.start_timestamp}-{current_segment.end_timestamp}")
                logger.warning(f"  -> Text: '{current_segment.text}'")
                logger.warning(f"  -> Language: {current_segment.language}")
                logger.warning(f"  -> Duration: {analysis.duration:.2f}s")
                logger.warning(f"  -> Context: {prev_segment.speaker} -> {current_segment.speaker} -> {next_segment.speaker}")
                
                segments_to_delete.append(current_segment)
                metrics.segments_marked_for_deletion += 1

        # Log detailed statistics
        metrics.analysis_time = time.time() - start_time
        _log_detection_statistics(metrics, detailed_analyses, segments)

        logger.info(f"Hallucination detection completed in {metrics.analysis_time:.3f}s")
        logger.info(f"Result: {len(segments_to_delete)} segments marked for deletion out of {len(segments)} total")

        return segments_to_delete

    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Hallucination detection failed after {total_time:.3f}s: {e}", exc_info=True)
        raise HallucinationDetectionError(f"Detection failed: {e}")


def get_hallucination_texts(db: Session) -> Set[str]:
    """
    Gets all active hallucination texts from the database with comprehensive logging.
    
    Args:
        db: Database session
        
    Returns:
        Set of texts in lowercase for detection
        
    Raises:
        HallucinationDetectionError: If database query fails
    """
    start_time = time.time()
    logger.debug("Querying database for active hallucination patterns")
    
    try:
        hallucination_rules = db.query(Hallucination).filter(Hallucination.is_active == True).all()
        
        query_time = time.time() - start_time
        logger.debug(f"Database query completed in {query_time:.3f}s")
        
        if not hallucination_rules:
            logger.warning("No active hallucination rules found in database")
            return set()
        
        # Process rules and create lowercase set
        hallucination_texts = set()
        language_counts = Counter()
        
        for rule in hallucination_rules:
            if rule.text_to_match:
                text_lower = rule.text_to_match.lower()
                hallucination_texts.add(text_lower)
                
                if rule.language:
                    language_counts[rule.language] += 1
                else:
                    language_counts['unspecified'] += 1
        
        logger.info(f"Loaded {len(hallucination_texts)} active hallucination patterns")
        logger.debug(f"Language distribution: {dict(language_counts)}")
        
        # Log some examples (first 5 patterns)
        if hallucination_texts:
            examples = list(hallucination_texts)[:5]
            logger.debug(f"Example patterns: {examples}")
            if len(hallucination_texts) > 5:
                logger.debug(f"... and {len(hallucination_texts) - 5} more patterns")
        
        return hallucination_texts

    except SQLAlchemyError as e:
        query_time = time.time() - start_time
        logger.error(f"Database error retrieving hallucination texts after {query_time:.3f}s: {e}", exc_info=True)
        raise HallucinationDetectionError(f"Database query failed: {e}")
    except Exception as e:
        query_time = time.time() - start_time
        logger.error(f"Unexpected error retrieving hallucination texts after {query_time:.3f}s: {e}", exc_info=True)
        raise HallucinationDetectionError(f"Failed to get hallucination texts: {e}")


def get_primary_languages(segments: List[Segment]) -> Set[str]:
    """
    Identifies primary languages based on frequency of appearance with detailed analysis.
    
    Args:
        segments: List of segments
        
    Returns:
        Set of languages that appear more than 2 times
        
    Raises:
        HallucinationDetectionError: If language analysis fails
    """
    start_time = time.time()
    logger.debug(f"Analyzing language distribution across {len(segments)} segments")
    
    try:
        if not segments:
            logger.warning("No segments provided for language analysis")
            return set()
        
        language_counts = Counter()
        unknown_count = 0
        none_count = 0
        
        # Count languages
        for segment in segments:
            if not segment.language:
                none_count += 1
            elif segment.language == 'unknown':
                unknown_count += 1
            else:
                language_counts[segment.language] += 1
        
        analysis_time = time.time() - start_time
        
        # Determine primary languages (threshold: > 2 occurrences)
        PRIMARY_LANGUAGE_THRESHOLD = 2
        primary_languages = {lang for lang, count in language_counts.items() if count > PRIMARY_LANGUAGE_THRESHOLD}
        
        # Log detailed statistics
        total_segments = len(segments)
        logger.info(f"Language analysis completed in {analysis_time:.3f}s:")
        logger.info(f"  - Total segments: {total_segments}")
        logger.info(f"  - Segments with None language: {none_count}")
        logger.info(f"  - Segments with 'unknown' language: {unknown_count}")
        logger.info(f"  - Detected languages: {len(language_counts)}")
        logger.info(f"  - Primary languages (>{PRIMARY_LANGUAGE_THRESHOLD} occurrences): {primary_languages}")
        
        # Log language distribution
        if language_counts:
            logger.debug("Language distribution:")
            for lang, count in language_counts.most_common():
                percentage = (count / total_segments) * 100
                is_primary = lang in primary_languages
                logger.debug(f"  - {lang}: {count} segments ({percentage:.1f}%) {'[PRIMARY]' if is_primary else ''}")
        
        # Warnings for unusual patterns
        if not primary_languages:
            logger.warning("No primary languages identified - all languages appear ≤2 times")
        
        if unknown_count > total_segments * 0.5:
            logger.warning(f"High proportion of unknown language segments: {unknown_count}/{total_segments} ({unknown_count/total_segments:.1%})")
        
        return primary_languages

    except Exception as e:
        analysis_time = time.time() - start_time
        logger.error(f"Language analysis failed after {analysis_time:.3f}s: {e}", exc_info=True)
        raise HallucinationDetectionError(f"Language analysis failed: {e}")


def _log_detection_statistics(
    metrics: HallucinationDetectionMetrics, 
    analyses: List[SegmentAnalysis], 
    segments: List[Segment]
) -> None:
    """
    Logs detailed statistics about the hallucination detection process.
    
    Args:
        metrics: Detection metrics
        analyses: List of detailed segment analyses
        segments: Original segments list
    """
    logger.info("=== HALLUCINATION DETECTION STATISTICS ===")
    logger.info(f"Total segments analyzed: {metrics.segments_analyzed}")
    logger.info(f"Sandwiched segments found: {metrics.sandwiched_segments_found}")
    logger.info(f"Blocklist matches: {metrics.blocklist_matches}")
    logger.info(f"Language outliers: {metrics.language_outliers}")
    logger.info(f"Segments marked for deletion: {metrics.segments_marked_for_deletion}")
    logger.info(f"Analysis time: {metrics.analysis_time:.3f}s")
    
    if metrics.segments_analyzed > 0:
        logger.info(f"Detection rate: {metrics.segments_marked_for_deletion/metrics.segments_analyzed:.1%} of analyzed segments")
    
    # Speaker-based statistics
    speaker_stats = defaultdict(lambda: {'total': 0, 'hallucinations': 0})
    
    for segment in segments:
        if segment.speaker:
            speaker_stats[segment.speaker]['total'] += 1
    
    for analysis in analyses:
        if analysis.matches_blocklist or analysis.is_language_outlier:
            speaker = analysis.segment.speaker
            if speaker:
                speaker_stats[speaker]['hallucinations'] += 1
    
    # Log speaker statistics
    if speaker_stats:
        logger.debug("Speaker-based hallucination statistics:")
        for speaker, stats in sorted(speaker_stats.items()):
            if stats['total'] > 0:
                rate = stats['hallucinations'] / stats['total']
                logger.debug(f"  - {speaker}: {stats['hallucinations']}/{stats['total']} ({rate:.1%}) hallucinations")
    
    # Log timing distribution for sandwiched segments
    sandwiched_analyses = [a for a in analyses if a.is_sandwiched]
    if sandwiched_analyses:
        durations = [a.duration for a in sandwiched_analyses]
        avg_duration = sum(durations) / len(durations)
        
        logger.debug(f"Sandwiched segment timing statistics:")
        logger.debug(f"  - Average duration: {avg_duration:.2f}s")
        logger.debug(f"  - Duration range: {min(durations):.2f}s - {max(durations):.2f}s")


def analyze_hallucination_patterns(segments_to_delete: List[Segment]) -> Dict[str, Any]:
    """
    Analyzes patterns in detected hallucinations for insights and reporting.
    
    Args:
        segments_to_delete: List of segments identified as hallucinations
        
    Returns:
        Dictionary containing pattern analysis results
    """
    if not segments_to_delete:
        logger.debug("No hallucinations to analyze")
        return {"total_hallucinations": 0}
    
    logger.debug(f"Analyzing patterns in {len(segments_to_delete)} detected hallucinations")
    
    # Analyze patterns
    speaker_distribution = Counter(seg.speaker for seg in segments_to_delete if seg.speaker)
    language_distribution = Counter(seg.language for seg in segments_to_delete if seg.language)
    text_lengths = [len(seg.text) if seg.text else 0 for seg in segments_to_delete]
    
    # Calculate timing statistics
    durations = []
    for seg in segments_to_delete:
        try:
            start = parse_timestamp_to_seconds(seg.start_timestamp)
            end = parse_timestamp_to_seconds(seg.end_timestamp)
            durations.append(end - start)
        except Exception:
            continue
    
    analysis_results = {
        "total_hallucinations": len(segments_to_delete),
        "speaker_distribution": dict(speaker_distribution),
        "language_distribution": dict(language_distribution),
        "average_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        "average_duration": sum(durations) / len(durations) if durations else 0,
        "duration_range": (min(durations), max(durations)) if durations else (0, 0)
    }
    
    logger.debug(f"Hallucination pattern analysis: {analysis_results}")
    return analysis_results