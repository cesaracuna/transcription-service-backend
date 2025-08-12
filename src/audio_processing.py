"""
Specialized functions for audio processing, transcription and diarization.
Enhanced with comprehensive logging, error handling, and performance monitoring.
"""

import os
import time
import torch
import librosa
import logging
import warnings
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pyannote.audio import Pipeline
from langdetect import detect, LangDetectException

from .constants import (
    WHISPER_MODEL_PATH, DIARIZATION_MODEL_PATH, SAMPLE_RATE, MIN_SEGMENT_LENGTH,
    VALID_LANGUAGES, MIN_DURATION_FOR_MERGE, MAX_PAUSE_FOR_MERGE,
    DIARIZATION_ONSET, DIARIZATION_OFFSET, DIARIZATION_MIN_DURATION_ON, DIARIZATION_MIN_DURATION_OFF,
    REPETITION_COMPRESSION_THRESHOLD, MAX_SEGMENT_DURATION, MIN_CHUNK_DURATION, 
    SILENCE_THRESHOLD, MIN_SILENCE_DURATION, CHUNK_OVERLAP, WORD_BOUNDARY_BUFFER,
    MAX_AUDIO_CHUNK_DURATION, MIN_SPEAKER_SEGMENT_DURATION,
    WHISPER_BEAM_SIZE, WHISPER_BATCH_SIZE, WHISPER_TEMPERATURE, WHISPER_NO_REPEAT_NGRAM_SIZE
)
from .utils import format_timestamp, is_silent, get_compression_ratio, remap_speakers_chronologically

# Create module-specific logger
logger = logging.getLogger(__name__)


@dataclass
class AudioProcessingMetrics:
    """Data class to track audio processing metrics."""
    audio_duration: float = 0.0
    diarization_time: float = 0.0
    transcription_time: float = 0.0
    segments_processed: int = 0
    segments_skipped_silence: int = 0
    segments_skipped_short: int = 0
    segments_repetitive: int = 0
    language_corrections: int = 0
    total_processing_time: float = 0.0


class AudioProcessingError(Exception):
    """Custom exception for audio processing errors."""
    pass


def validate_audio_file(audio_path: str) -> None:
    """
    Validates that the audio file exists and is accessible.
    
    Args:
        audio_path: Path to the audio file
        
    Raises:
        AudioProcessingError: If file doesn't exist or isn't accessible
    """
    if not os.path.exists(audio_path):
        error_msg = f"Audio file not found: {audio_path}"
        logger.error(error_msg)
        raise AudioProcessingError(error_msg)
    
    if not os.access(audio_path, os.R_OK):
        error_msg = f"Audio file not readable: {audio_path}"
        logger.error(error_msg)
        raise AudioProcessingError(error_msg)
    
    file_size = os.path.getsize(audio_path)
    logger.debug(f"Audio file validated: {audio_path} (Size: {file_size} bytes)")


def initialize_ai_models(device: str) -> Tuple[WhisperProcessor, WhisperForConditionalGeneration, Pipeline]:
    """
    Initializes Whisper and diarization models with comprehensive error handling and logging.
    
    Args:
        device: Computing device ("cuda:0" or "cpu")
        
    Returns:
        Tuple with (processor, whisper_model, diarization_pipeline)
        
    Raises:
        AudioProcessingError: If model initialization fails
    """
    start_time = time.time()
    logger.info(f"Starting AI model initialization on device: {device}")
    
    try:
        # Initial configuration
        logger.debug("Configuring PyTorch backend settings")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        warnings.filterwarnings("ignore", category=UserWarning)

        # Validate device availability
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(f"CUDA device {device} requested but not available. Falling back to CPU.")
            device = "cpu"

        # Hugging Face token validation
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN environment variable not set. Diarization might fail.")
        else:
            logger.debug("HF_TOKEN found and will be used for model authentication")

        # Load Whisper model
        logger.info(f"Loading Whisper model: {WHISPER_MODEL_PATH}")
        whisper_start = time.time()
        
        try:
            whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
            whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH).to(device)
            # Configure generation settings using modern parameters
            whisper_model.generation_config.task = "transcribe"
            whisper_model.generation_config.forced_decoder_ids = None
            
            whisper_time = time.time() - whisper_start
            logger.info(f"Whisper model loaded successfully in {whisper_time:.2f}s. Model size: {sum(p.numel() for p in whisper_model.parameters()):,} parameters")
            logger.info(f"Whisper config: beam_size={WHISPER_BEAM_SIZE}, batch_size={WHISPER_BATCH_SIZE}, temp={WHISPER_TEMPERATURE}, no_repeat_ngram={WHISPER_NO_REPEAT_NGRAM_SIZE}")
            
            # Test Whisper model
            logger.debug("Testing Whisper model with dummy input")
            dummy_audio = torch.randn(1, 16000).to(device)  # 1 second of dummy audio
            with torch.no_grad():
                inputs = whisper_processor(dummy_audio.squeeze().cpu().numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
                _ = whisper_model.generate(**inputs, max_length=10)
            logger.debug("Whisper model test successful")
            
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
            raise AudioProcessingError(f"Whisper model initialization failed: {e}")

        # Load Diarization pipeline
        logger.info(f"Loading diarization pipeline: {DIARIZATION_MODEL_PATH}")
        diarization_start = time.time()
        
        try:
            diarization_pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL_PATH, use_auth_token=hf_token)
            diarization_pipeline.to(torch.device(device))
            
            # Configure diarization parameters
            diarization_pipeline.segmentation.onset = DIARIZATION_ONSET
            diarization_pipeline.segmentation.offset = DIARIZATION_OFFSET
            diarization_pipeline.segmentation.min_duration_on = DIARIZATION_MIN_DURATION_ON
            diarization_pipeline.segmentation.min_duration_off = DIARIZATION_MIN_DURATION_OFF
            
            diarization_time = time.time() - diarization_start
            logger.info(f"Diarization pipeline loaded successfully in {diarization_time:.2f}s")
            logger.info(f"Diarization params: onset={DIARIZATION_ONSET}, offset={DIARIZATION_OFFSET}, "
                       f"min_on={DIARIZATION_MIN_DURATION_ON}, min_off={DIARIZATION_MIN_DURATION_OFF}")
            
        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}", exc_info=True)
            raise AudioProcessingError(f"Diarization pipeline initialization failed: {e}")

        total_time = time.time() - start_time
        logger.info(f"AI model initialization completed successfully in {total_time:.2f}s "
                   f"(Whisper: {whisper_time:.2f}s, Diarization: {diarization_time:.2f}s)")

        return whisper_processor, whisper_model, diarization_pipeline

    except AudioProcessingError:
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Unexpected error during model initialization after {total_time:.2f}s: {e}", exc_info=True)
        raise AudioProcessingError(f"Model initialization failed: {e}")


def load_and_validate_audio(audio_path: str) -> Tuple[np.ndarray, int, float]:
    """
    Loads and validates audio file with comprehensive logging.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Tuple of (audio_waveform, sampling_rate, duration_seconds)
        
    Raises:
        AudioProcessingError: If audio loading fails
    """
    logger.info(f"Loading audio file: {audio_path}")
    start_time = time.time()
    
    try:
        validate_audio_file(audio_path)
        
        # Load audio with librosa
        audio_waveform, sampling_rate = librosa.load(audio_path, sr=SAMPLE_RATE)
        duration_in_seconds = librosa.get_duration(y=audio_waveform, sr=sampling_rate)
        
        load_time = time.time() - start_time
        
        # Log audio characteristics
        logger.info(f"  - Audio loaded successfully in {load_time:.2f}s:")
        logger.info(f"    • Duration: {duration_in_seconds:.2f} seconds")
        logger.info(f"    • Sample rate: {sampling_rate} Hz")
        logger.info(f"    • Samples: {len(audio_waveform):,}")
        logger.info(f"    • RMS level: {np.sqrt(np.mean(audio_waveform**2)):.4f}")
        logger.info(f"    • Peak level: {np.max(np.abs(audio_waveform)):.4f}")
        
        # Validate audio quality
        if duration_in_seconds < 0.1:
            logger.warning(f"Audio file is very short: {duration_in_seconds:.2f}s")
            return audio_waveform, sampling_rate, duration_in_seconds
        elif duration_in_seconds > 3600:  # 1 hour
            logger.warning(f"Audio file is very long: {duration_in_seconds:.2f}s - processing may take a while")
        
        if np.max(np.abs(audio_waveform)) < 0.001:
            logger.warning("Audio appears to have very low volume")
        
        return audio_waveform, sampling_rate, duration_in_seconds
        
    except AudioProcessingError:
        raise
    except Exception as e:
        load_time = time.time() - start_time
        logger.error(f"Failed to load audio file after {load_time:.2f}s: {e}", exc_info=True)
        raise AudioProcessingError(f"Audio loading failed: {e}")


def run_diarization(audio_waveform: np.ndarray, sampling_rate: int, diarization_pipeline: Pipeline) -> Any:
    """
    Runs speaker diarization with comprehensive logging and error handling.
    
    Args:
        audio_waveform: Audio waveform data
        sampling_rate: Audio sampling rate
        diarization_pipeline: Configured diarization pipeline
        
    Returns:
        Diarization result
        
    Raises:
        AudioProcessingError: If diarization fails
    """
    logger.info("Starting speaker diarization...")
    start_time = time.time()
    
    try:
        # Convert audio to PyTorch tensor
        audio_tensor = torch.from_numpy(audio_waveform).unsqueeze(0)
        logger.debug(f"Audio tensor shape: {audio_tensor.shape}")
        
        # Run diarization
        diarization = diarization_pipeline({
            "waveform": audio_tensor, 
            "sample_rate": sampling_rate
        })
        
        diarization_time = time.time() - start_time
        
        # Log diarization results
        if not diarization.get_timeline():
            logger.warning("Diarization pipeline returned no speaker segments")
            return diarization
        
        # Count segments and speakers
        segments = list(diarization.itertracks(yield_label=True))
        speakers = set(speaker for _, _, speaker in segments)
        total_speech_time = sum(turn.end - turn.start for turn, _, _ in segments)
        
        logger.info(f"  - Diarization completed in {diarization_time:.2f}s:")
        logger.info(f"    • Found {len(speakers)} unique speakers")
        logger.info(f"    • Generated {len(segments)} segments")
        logger.info(f"    • Total speech time: {total_speech_time:.2f}s")
        logger.info(f"    • Speech ratio: {total_speech_time/len(audio_waveform)*sampling_rate:.1%}")
        
        # Log detailed segment information
        _log_diarization_output(diarization)
        
        return diarization
        
    except Exception as e:
        diarization_time = time.time() - start_time
        logger.error(f"Diarization failed after {diarization_time:.2f}s: {e}", exc_info=True)
        raise AudioProcessingError(f"Diarization failed: {e}")


def transcribe_segment(
    segment_waveform: np.ndarray, 
    sampling_rate: int, 
    whisper_processor: WhisperProcessor, 
    whisper_model: WhisperForConditionalGeneration, 
    device: str,
    speaker: str,
    start_time: float,
    end_time: float,
    speaker_language: Dict[str, str],
    metrics: AudioProcessingMetrics,
    is_chunked_segment: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Transcribes a single audio segment with comprehensive logging and validation.
    
    Args:
        segment_waveform: Audio segment data
        sampling_rate: Audio sampling rate
        whisper_processor: Whisper processor
        whisper_model: Whisper model
        device: Computing device
        speaker: Speaker ID
        start_time: Segment start time
        end_time: Segment end time
        speaker_language: Dictionary tracking speaker languages
        metrics: Processing metrics tracker
        
    Returns:
        Transcribed segment data or None if segment is skipped
    """
    duration = end_time - start_time
    segment_start_time = time.time()
    
    logger.debug(f"Processing segment {speaker}: {start_time:.2f}s-{end_time:.2f}s ({duration:.2f}s)")
    
    try:
        # 1. Validate segment length
        if len(segment_waveform) < MIN_SEGMENT_LENGTH:
            logger.debug(f"    -> SKIPPED: Segment too short ({len(segment_waveform)} samples < {MIN_SEGMENT_LENGTH})")
            metrics.segments_skipped_short += 1
            return None

        # 2. Silence detection
        rms_value, is_silent_flag = is_silent(segment_waveform)
        if is_silent_flag:
            logger.debug(f"    -> SKIPPED: Silent segment (RMS: {rms_value:.4f})")
            metrics.segments_skipped_silence += 1
            return None

        # 3. Transcription
        logger.debug(f"    -> Transcribing segment (RMS: {rms_value:.4f})")
        transcription_start = time.time()
        
        inputs = whisper_processor(segment_waveform, sampling_rate=sampling_rate, return_tensors="pt").to(device)
        
        with torch.no_grad():
            predicted_ids = whisper_model.generate(
                **inputs, 
                task="transcribe", 
                temperature=WHISPER_TEMPERATURE,
                num_beams=WHISPER_BEAM_SIZE,
                no_repeat_ngram_size=WHISPER_NO_REPEAT_NGRAM_SIZE
            )
        
        transcription_pass = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        transcription_time = time.time() - transcription_start
        
        logger.debug(f"    -> Transcription completed in {transcription_time:.3f}s: '{transcription_pass[:50]}{'...' if len(transcription_pass) > 50 else ''}'")

        # 4. Language detection
        detected_lang = 'unknown'
        if transcription_pass:
            try:
                detected_lang = detect(transcription_pass)
                logger.debug(f"    -> Detected language: {detected_lang}")
            except LangDetectException:
                logger.debug("    -> Language detection failed, using 'unknown'")
                detected_lang = 'unknown'

        final_transcription = transcription_pass
        final_segment_lang = detected_lang
        last_valid_lang = speaker_language.get(speaker)

        # 5. Language correction if necessary
        if detected_lang not in VALID_LANGUAGES and last_valid_lang:
            logger.debug(f"    -> Language correction: {detected_lang} -> {last_valid_lang} for {speaker}")
            
            correction_start = time.time()
            
            with torch.no_grad():
                corrected_ids = whisper_model.generate(
                    **inputs,
                    task="transcribe",
                    language=last_valid_lang,
                    temperature=WHISPER_TEMPERATURE,
                    num_beams=WHISPER_BEAM_SIZE,
                    no_repeat_ngram_size=WHISPER_NO_REPEAT_NGRAM_SIZE
                )
            
            final_transcription = whisper_processor.batch_decode(corrected_ids, skip_special_tokens=True)[0].strip()
            final_segment_lang = last_valid_lang
            correction_time = time.time() - correction_start
            
            logger.debug(f"    -> Language correction completed in {correction_time:.3f}s")
            metrics.language_corrections += 1
            
        elif detected_lang in VALID_LANGUAGES:
            if last_valid_lang != detected_lang:
                logger.debug(f"    -> Language update for {speaker}: {last_valid_lang} -> {detected_lang}")
            speaker_language[speaker] = detected_lang

        # 6. Repetition detection
        compression_ratio = get_compression_ratio(final_transcription)
        if compression_ratio < REPETITION_COMPRESSION_THRESHOLD and len(final_transcription) > 20:
            logger.debug(f"    -> REPETITIVE content detected (ratio: {compression_ratio:.2f})")
            final_transcription = "[Repetitive content discarded]"
            metrics.segments_repetitive += 1

        segment_processing_time = time.time() - segment_start_time
        logger.debug(f"    -> Segment processed in {segment_processing_time:.3f}s")
        
        metrics.segments_processed += 1
        
        # Clean text duplication only for chunked segments (where overlaps can occur)
        cleaned_transcription = final_transcription
        if is_chunked_segment:
            cleaned_transcription = remove_text_duplication(final_transcription)
            if cleaned_transcription != final_transcription:
                logger.debug(f"    -> Applied duplication cleanup to chunked segment")
        
        return {
            "start": format_timestamp(start_time), 
            "end": format_timestamp(end_time),
            "speaker": speaker, 
            "text": cleaned_transcription, 
            "language": final_segment_lang,
            "processing_time": segment_processing_time,
            "transcription_time": transcription_time,
            "rms_level": rms_value,
            "compression_ratio": compression_ratio
        }

    except Exception as e:
        segment_processing_time = time.time() - segment_start_time
        logger.error(f"Error processing segment {speaker} ({start_time:.2f}s-{end_time:.2f}s) "
                    f"after {segment_processing_time:.3f}s: {e}", exc_info=True)
        return None


# # def transcribe_segments_batch(segments_batch: List[Tuple[np.ndarray, Dict[str, Any]]], 
# #                              sampling_rate: int, 
# #                              whisper_processor: WhisperProcessor, 
# #                              whisper_model: WhisperForConditionalGeneration, 
# #                              device: str) -> List[Dict[str, Any]]:
# #     """
# #     Transcribes multiple segments in batch for improved efficiency.
    
# #     Args:
# #         segments_batch: List of tuples (segment_waveform, segment_info)
# #         sampling_rate: Audio sampling rate
# #         whisper_processor: Whisper processor
# #         whisper_model: Whisper model
# #         device: Computing device
        
# #     Returns:
# #         List of transcription results
# #     """
# #     if not segments_batch:
# #         return []
    
# #     logger.debug(f"Processing batch of {len(segments_batch)} segments")
    
# #     try:
# #         # Prepare batch inputs
# #         audio_arrays = []
# #         segment_infos = []
        
# #         for segment_waveform, segment_info in segments_batch:
# #             audio_arrays.append(segment_waveform)
# #             segment_infos.append(segment_info)
        
# #         # Process batch (currently Whisper doesn't support true batch processing well)
# #         # So we process individually but with optimized parameters
# #         results = []
        
# #         for i, (segment_waveform, segment_info) in enumerate(segments_batch):
# #             inputs = whisper_processor(segment_waveform, sampling_rate=sampling_rate, return_tensors="pt").to(device)
            
# #             with torch.no_grad():
# #                 predicted_ids = whisper_model.generate(
# #                     **inputs, 
# #                     task="transcribe", 
# #                     temperature=WHISPER_TEMPERATURE,
# #                     num_beams=WHISPER_BEAM_SIZE,
# #                     no_repeat_ngram_size=WHISPER_NO_REPEAT_NGRAM_SIZE,
# #                     max_length=224  # Whisper max tokens
# #                 )
            
# #             transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            
# #             if transcription:
# #                 results.append({
# #                     "start": segment_info["start"], 
# #                     "end": segment_info["end"],
# #                     "speaker": segment_info["speaker"], 
# #                     "text": transcription, 
# #                     "language": "auto",  # Will be detected later
# #                 })
        
# #         logger.debug(f"Batch processing completed: {len(results)} successful transcriptions")
# #         return results
        
# #     except Exception as e:
# #         logger.error(f"Batch transcription failed: {e}", exc_info=True)
# #         # Fallback to individual processing
# #         return []


def process_audio_pipeline(
    audio_path: str, 
    whisper_processor: WhisperProcessor, 
    whisper_model: WhisperForConditionalGeneration, 
    diarization_pipeline: Pipeline, 
    device: str,
    db_session=None,
    job_instance=None
) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    """
    Complete audio processing pipeline with diarization, chunking, transcription, and metrics.
    
    Args:
        audio_path: Path to audio file
        whisper_processor: Whisper processor
        whisper_model: Whisper model
        diarization_pipeline: Diarization pipeline
        device: Computing device
        
    Returns:
        Tuple with (final_segments, duration_in_seconds)
        
    Raises:
        AudioProcessingError: If transcription fails
    """
    total_start_time = time.time()
    metrics = AudioProcessingMetrics()
    
    logger.info(f"Starting transcription pipeline for: {audio_path}")
    
    try:
        # Phase 1: Audio Loading and Validation
        logger.info("Phase 1: Audio Loading and Validation")
        audio_waveform, sampling_rate, duration_in_seconds = load_and_validate_audio(audio_path)
        metrics.audio_duration = duration_in_seconds

        # Phase 2: Speaker Diarization + Database Insertion + Status 'in progress'
        logger.info("Phase 2: Speaker Diarization")
        diarization_start = time.time()
        diarization = run_diarization(audio_waveform, sampling_rate, diarization_pipeline)
        metrics.diarization_time = time.time() - diarization_start

        # Extract raw diarization segments
        raw_diarization_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw_diarization_segments.append({
                'start': turn.start,
                'end': turn.end,
                'speaker': speaker
            })
        
        logger.info(f"  - Extracted {len(raw_diarization_segments)} raw diarization segments")
        
        # Insert diarization segments into database and mark as 'en proceso'
        if db_session and job_instance:
            from .tasks import save_diarization_segments_with_status
            diarization_saved = save_diarization_segments_with_status(
                db_session, job_instance, raw_diarization_segments, status='processing'
            )
            logger.info(f"  - Saved {diarization_saved} diarization segments to database with status 'processing'")

        # Phase 3: Overlapping Diarization Segments Cleanup
        logger.info("Phase 3: Overlapping Diarization Segments Cleanup")
        cleanup_start = time.time()
        cleaned_segments = clean_overlapping_diarization_segments(raw_diarization_segments)
        cleanup_time = time.time() - cleanup_start
        logger.info(f"  - Segment cleanup completed in {cleanup_time:.2f}s")
        # TEMP
        # cleaned_segments = raw_diarization_segments

        # Phase 4: Clean Diarization Segments Merging and Chunking
        logger.info("Phase 4: Speaker Intervention Chunking (max 28min per chunk)")
        merge_start = time.time()
        merged_turns = merge_diarization_segments(
            cleaned_segments, 
            audio_waveform=audio_waveform, 
            sampling_rate=sampling_rate,
            audio_path=audio_path,
            job_id=str(job_instance.id) if job_instance else None
        )
        merge_time = time.time() - merge_start
        
        logger.info(f"  - Speaker intervention chunking completed in {merge_time:.2f}s. Final segments: {len(merged_turns)}")

        # Phase 5: Individual Long Segment Processing
        logger.info("Phase 5: Individual Long Segment Processing")
        
        # 5a. Detect silence intervals for intelligent long segment division  
        silence_start = time.time()
        audio_silence_intervals = detect_silence_intervals(audio_waveform, sampling_rate)
        silence_time = time.time() - silence_start
        logger.info(f"  - Silence detection completed in {silence_time:.2f}s. Found {len(audio_silence_intervals)} silence intervals")
        
        # Note: Audio chunking (28min max) was already completed in Phase 4
        # Phase 5 only handles individual segments that exceed MAX_SEGMENT_DURATION
        audio_chunks = [merged_turns]  # Single chunk with all segments from Phase 4

        # Phase 6: Segment Transcription
        logger.info("Phase 6: Segment Transcription")
        logger.info(f"  - Using Whisper config: beam_size={WHISPER_BEAM_SIZE}, batch_size={WHISPER_BATCH_SIZE}, temp={WHISPER_TEMPERATURE}")
        transcription_start = time.time()
        
        all_final_segments = []
        total_long_segments_count = 0
        total_chunks_created = 0
        
        for chunk_idx, chunk_segments in enumerate(audio_chunks):
            logger.info(f"  Processing audio chunk {chunk_idx + 1}/{len(audio_chunks)} ({len(chunk_segments)} segments)")
            
            # 6a. Process long segments within each chunk
            chunk_processing_start = time.time()
            
            final_segments_for_transcription = []
            long_segments_count = 0
            chunks_created = 0
            
            for segment in chunk_segments:
                duration = segment['end'] - segment['start']
                
                if duration > MAX_SEGMENT_DURATION:
                    logger.debug(f"    • Long segment detected: {segment['speaker']} ({duration:.2f}s) - chunking...")
                    long_segments_count += 1
                    
                    # Chunk the segment
                    chunks = chunk_long_segment(audio_waveform, sampling_rate, segment, audio_silence_intervals)
                    final_segments_for_transcription.extend(chunks)
                    chunks_created += len(chunks)
                    
                else:
                    # Keep segment as is
                    final_segments_for_transcription.append(segment)
            
            chunk_processing_time = time.time() - chunk_processing_start
            logger.info(f"    • Segment processing completed in {chunk_processing_time:.2f}s: {long_segments_count} long segments → {chunks_created} chunks")
            
            all_final_segments.extend(final_segments_for_transcription)
            total_long_segments_count += long_segments_count
            total_chunks_created += chunks_created
        
        # 6b. Transcribe all segments
        final_segments = []
        speaker_language = {}
        
        for i, turn in enumerate(all_final_segments):
            start_time, end_time, speaker = turn['start'], turn['end'], turn['speaker']
            
            logger.debug(f"Transcribing segment {i+1}/{len(all_final_segments)}: {speaker}")
            
            # Extract segment audio
            segment_waveform = audio_waveform[int(start_time * sampling_rate):int(end_time * sampling_rate)]
            
            # Check if this segment came from chunking (has overlap risk)
            is_chunked = turn.get('original_segment', False)
            
            # Transcribe segment
            segment_result = transcribe_segment(
                segment_waveform, sampling_rate, whisper_processor, whisper_model, device,
                speaker, start_time, end_time, speaker_language, metrics, is_chunked
            )
            
            if segment_result:
                final_segments.append(segment_result)
        
        metrics.transcription_time = time.time() - transcription_start
        logger.info(f"  - Transcription completed: {len(final_segments)} segments transcribed")

        # Phase 7: Speaker Labels Reassignment
        logger.info("Phase 7: Speaker Labels Reassignment")
        remap_start = time.time()
        remapped_segments = remap_speakers_chronologically(final_segments)
        remap_time = time.time() - remap_start
        
        logger.info(f"  - Speaker remapping completed in {remap_time:.2f}s")

        # Phase 8: Final Metrics and Logging
        logger.info("Phase 8: Final Metrics and Logging")
        metrics.total_processing_time = time.time() - total_start_time
        
        logger.info("=== TRANSCRIPTION PIPELINE COMPLETED ===")
        logger.info(f"  - Total processing time: {metrics.total_processing_time:.2f}s")
        logger.info(f"  - Audio duration: {metrics.audio_duration:.2f}s")
        logger.info(f"  - Processing speed: {metrics.audio_duration/metrics.total_processing_time:.1f}x realtime")
        logger.info(f"  - Diarization time: {metrics.diarization_time:.2f}s")
        logger.info(f"  - Transcription time: {metrics.transcription_time:.2f}s")
        logger.info(f"  - Segments processed: {metrics.segments_processed}")
        logger.info(f"  - Segments skipped (silence): {metrics.segments_skipped_silence}")
        logger.info(f"  - Segments skipped (short): {metrics.segments_skipped_short}")
        logger.info(f"  - Repetitive segments: {metrics.segments_repetitive}")
        logger.info(f"  - Language corrections: {metrics.language_corrections}")
        logger.info(f"  - Final segments: {len(remapped_segments)}")
        logger.info(f"  - Long segments chunked: {total_long_segments_count}")
        logger.info(f"  - Total segment chunks created: {total_chunks_created}")
        
        # Performance metrics with Whisper config impact
        avg_transcription_per_segment = metrics.transcription_time / max(metrics.segments_processed, 1)
        transcription_rtf = metrics.transcription_time / metrics.audio_duration  # Real-time factor
        logger.info(f"  - Avg transcription per segment: {avg_transcription_per_segment:.3f}s")
        logger.info(f"  - Transcription RTF: {transcription_rtf:.2f}x (beam={WHISPER_BEAM_SIZE})")
        
        # Performance recommendations based on RTF
        if transcription_rtf > 0.5:
            logger.warning(f"  ⚠️  High RTF detected. Consider reducing beam_size from {WHISPER_BEAM_SIZE} for faster processing")
        elif transcription_rtf < 0.1:
            logger.info(f"  ✅ Excellent performance. Could increase beam_size from {WHISPER_BEAM_SIZE} for better quality")
        else:
            logger.info(f"  ✓ Good performance. RTF {transcription_rtf:.2f}x is within optimal range (0.1-0.5x)")
        
        # Log language distribution
        language_counts = {}
        for segment in remapped_segments:
            lang = segment.get('language', 'unknown')
            language_counts[lang] = language_counts.get(lang, 0) + 1
        
        logger.info(f"  - Language distribution: {dict(sorted(language_counts.items()))}")

        return remapped_segments, duration_in_seconds, merged_turns

    except AudioProcessingError:
        raise
    except Exception as e:
        total_time = time.time() - total_start_time
        logger.error(f"Transcription pipeline failed after {total_time:.2f}s: {e}", exc_info=True)
        raise AudioProcessingError(f"Transcription failed: {e}")


def _log_diarization_output(diarization) -> None:
    """
    Logs detailed diarization output for debugging with enhanced information.
    
    Args:
        diarization: Result from diarization pipeline
    """
    logger.debug("--- RAW DIARIZATION OUTPUT ---")
    
    if not diarization.get_timeline():
        logger.warning("Diarization pipeline returned no speaker segments")
    else:
        segments = list(diarization.itertracks(yield_label=True))
        speakers = {}
        
        for i, (turn, _, speaker) in enumerate(segments):
            duration = turn.end - turn.start
            
            if speaker not in speakers:
                speakers[speaker] = {'count': 0, 'total_time': 0.0}
            
            speakers[speaker]['count'] += 1
            speakers[speaker]['total_time'] += duration
            
            if i < 10:  # Log first 10 segments in detail
                logger.debug(f"  -> Segment {i+1}: [Speaker: {speaker}] "
                           f"{turn.start:.2f}s-{turn.end:.2f}s ({duration:.2f}s)")
            elif i == 10:
                logger.debug(f"  -> ... and {len(segments) - 10} more segments")
        
        # Log speaker summary
        logger.debug("Speaker summary:")
        for speaker, stats in speakers.items():
            logger.debug(f"  -> {speaker}: {stats['count']} segments, {stats['total_time']:.2f}s total")
    
    logger.debug("--- END OF RAW DIARIZATION OUTPUT ---")


def merge_diarization_segments(segments: List[Dict[str, Any]], 
                             audio_waveform: np.ndarray = None, 
                             sampling_rate: int = None,
                             audio_path: str = None,
                             job_id: str = None) -> List[Dict[str, Any]]:
    """
    Groups consecutive segments by speaker and creates audio chunks using greedy algorithm.
    
    Process:
    1. Groups segments by speaker interventions (complete interventions only)
    2. Applies greedy algorithm to fill chunks with maximum 28 minutes
    3. Creates physical audio files for each chunk
    4. Ensures complete speaker interventions are never split
    
    Example:
    - Intervention A: 10min (complete) 
    - Intervention B: 15min (complete)
    - Intervention C: 10min (complete)
    
    Result:
    - Chunk 1: [A complete + B complete] = 25min ✓
    - Chunk 2: [C complete + next speaker...] = next chunk
    
    Args:
        segments: List of segment dictionaries
        audio_waveform: Complete audio waveform (for creating chunk files)  
        sampling_rate: Audio sampling rate
        audio_path: Original audio file path (for naming chunks)
        
    Returns:
        List of all segments (from all chunks combined)
    """
    if not segments:
        return []
    
    logger.info("Starting speaker intervention chunking with greedy algorithm...")
    start_time = time.time()
    
    # Step 1: Group consecutive segments by speaker interventions
    logger.info("Step 1: Grouping consecutive segments by speaker interventions")
    sorted_segments = sorted(segments, key=lambda x: x['start'])
    
    speaker_interventions = []
    current_speaker = None
    current_segments = []
    
    for segment in sorted_segments:
        if segment['speaker'] != current_speaker:
            # Finalize previous intervention
            if current_segments:
                intervention_start = current_segments[0]['start'] 
                intervention_end = current_segments[-1]['end']
                intervention_duration = intervention_end - intervention_start
                
                speaker_interventions.append({
                    'speaker': current_speaker,
                    'segments': current_segments.copy(),
                    'start': intervention_start,
                    'end': intervention_end, 
                    'duration': intervention_duration,
                    'segment_count': len(current_segments)
                })
                
                logger.debug(f"  - Intervention: {current_speaker} ({intervention_duration/60:.1f}min, {len(current_segments)} segments)")
            
            # Start new intervention
            current_speaker = segment['speaker']
            current_segments = [segment]
        else:
            # Continue current intervention
            current_segments.append(segment)
    
    # Add final intervention
    if current_segments:
        intervention_start = current_segments[0]['start']
        intervention_end = current_segments[-1]['end']
        intervention_duration = intervention_end - intervention_start
        
        speaker_interventions.append({
            'speaker': current_speaker,
            'segments': current_segments.copy(), 
            'start': intervention_start,
            'end': intervention_end,
            'duration': intervention_duration,
            'segment_count': len(current_segments)
        })
        
        logger.debug(f"  - Intervention: {current_speaker} ({intervention_duration/60:.1f}min, {len(current_segments)} segments)")
    
    logger.info(f"  - Created {len(speaker_interventions)} speaker interventions")
    
    # Step 2: Apply greedy algorithm to create chunks (max 28 minutes)
    logger.info(f"Step 2: Applying greedy algorithm (max {MAX_SEGMENT_DURATION}min per chunk)")
    MAX_CHUNK_SECONDS = MAX_SEGMENT_DURATION * 60  # 28 minutes in seconds
    
    chunks = []
    current_chunk_interventions = []
    current_chunk_duration = 0.0
    
    for intervention in speaker_interventions:
        intervention_duration = intervention['duration']
        
        logger.debug(f"  - Processing {intervention['speaker']}: {intervention_duration/60:.1f}min")
        
        # Check if this complete intervention fits in current chunk
        if current_chunk_duration + intervention_duration <= MAX_CHUNK_SECONDS:
            # Add complete intervention to current chunk
            current_chunk_interventions.append(intervention)
            current_chunk_duration += intervention_duration
            
            logger.debug(f"    ✓ Added to current chunk (total: {current_chunk_duration/60:.1f}min)")
        else:
            # Complete intervention doesn't fit, finalize current chunk
            if current_chunk_interventions:
                chunk_segments = []
                for interv in current_chunk_interventions:
                    chunk_segments.extend(interv['segments'])
                
                chunk_start = min(seg['start'] for seg in chunk_segments)
                chunk_end = max(seg['end'] for seg in chunk_segments)
                actual_duration = chunk_end - chunk_start
                
                chunks.append({
                    'segments': chunk_segments,
                    'interventions': current_chunk_interventions.copy(),
                    'start': chunk_start,
                    'end': chunk_end,
                    'duration': actual_duration,
                    'chunk_index': len(chunks) + 1
                })
                
                logger.info(f"  ✓ Finalized Chunk {len(chunks)}: "
                           f"{chunk_start/60:.1f}-{chunk_end/60:.1f}min "
                           f"({actual_duration/60:.1f}min, {len(current_chunk_interventions)} interventions, "
                           f"{len(chunk_segments)} segments)")
            
            # Start new chunk with current intervention  
            current_chunk_interventions = [intervention]
            current_chunk_duration = intervention_duration
            
            logger.debug(f"    → Started new chunk with {intervention['speaker']} ({intervention_duration/60:.1f}min)")
    
    # Add final chunk if not empty
    if current_chunk_interventions:
        chunk_segments = []
        for interv in current_chunk_interventions:
            chunk_segments.extend(interv['segments'])
        
        chunk_start = min(seg['start'] for seg in chunk_segments)
        chunk_end = max(seg['end'] for seg in chunk_segments)
        actual_duration = chunk_end - chunk_start
        
        chunks.append({
            'segments': chunk_segments,
            'interventions': current_chunk_interventions.copy(),
            'start': chunk_start,
            'end': chunk_end,
            'duration': actual_duration,
            'chunk_index': len(chunks) + 1
        })
        
        logger.info(f"  ✓ Finalized Final Chunk {len(chunks)}: "
                   f"{chunk_start/60:.1f}-{chunk_end/60:.1f}min "
                   f"({actual_duration/60:.1f}min, {len(current_chunk_interventions)} interventions, "
                   f"{len(chunk_segments)} segments)")
    
    # Step 3: Create physical audio files for each chunk  
    if audio_waveform is not None and sampling_rate is not None and audio_path is not None:
        logger.info("Step 3: Creating physical audio files for chunks")
        created_files = create_chunk_audio_files(chunks, audio_waveform, sampling_rate, audio_path, job_id)
        logger.info(f"  ✓ Created {len(created_files)} audio chunk files")
        
        # Clean up chunk files after processing
        cleanup_chunk_files(created_files)
    
    # Step 4: Return all segments combined
    all_segments = []
    for chunk in chunks:
        all_segments.extend(chunk['segments'])
    
    merge_time = time.time() - start_time
    logger.info("Greedy speaker intervention chunking completed:")
    logger.info(f"  - Processing time: {merge_time:.3f}s")
    logger.info(f"  - Original segments: {len(sorted_segments)}")
    logger.info(f"  - Speaker interventions: {len(speaker_interventions)}")  
    logger.info(f"  - Audio chunks created: {len(chunks)}")
    logger.info(f"  - Final segments: {len(all_segments)}")
    
    # Log chunk summary
    total_chunk_duration = sum(chunk['duration'] for chunk in chunks)
    logger.info(f"  - Total chunk duration: {total_chunk_duration/60:.1f}min")
    logger.info(f"  - Avg chunk duration: {total_chunk_duration/len(chunks)/60:.1f}min" if chunks else "  - No chunks created")
    
    return all_segments


def create_chunk_audio_files(chunks: List[Dict[str, Any]], 
                           audio_waveform: np.ndarray, 
                           sampling_rate: int, 
                           original_audio_path: str,
                           job_id: str = None) -> List[str]:
    """
    Creates physical audio files for each chunk in the chunks folder.
    
    Args:
        chunks: List of chunk dictionaries with start/end times
        audio_waveform: Complete original audio waveform
        sampling_rate: Audio sampling rate
        original_audio_path: Path to original audio file (for naming)
        
    Returns:
        List of created audio file paths
    """
    import soundfile as sf
    
    # Create chunks directory in correct location
    base_dir = os.path.dirname(original_audio_path)
    
    # Check if we're already in audio_files folder
    if os.path.basename(base_dir) == "audio_files":
        # Already in audio_files, just add chunks
        chunks_dir = os.path.join(base_dir, "chunks")
    else:
        # Need to create audio_files/chunks
        audio_files_dir = os.path.join(base_dir, "audio_files") 
        chunks_dir = os.path.join(audio_files_dir, "chunks")
    
    os.makedirs(chunks_dir, exist_ok=True)
    logger.debug(f"Created chunks directory: {chunks_dir}")
    
    # Get base filename without extension
    original_filename = os.path.splitext(os.path.basename(original_audio_path))[0]
    created_files = []
    
    logger.info(f"Creating {len(chunks)} audio chunk files...")
    
    for chunk in chunks:
        chunk_index = chunk['chunk_index']
        chunk_start = chunk['start']
        chunk_end = chunk['end']
        
        # Extract audio segment for this chunk
        start_sample = int(chunk_start * sampling_rate)
        end_sample = int(chunk_end * sampling_rate)
        
        # Ensure we don't go beyond audio boundaries
        start_sample = max(0, start_sample)
        end_sample = min(len(audio_waveform), end_sample)
        
        chunk_audio = audio_waveform[start_sample:end_sample]
        
        # Create filename with simplified format
        if job_id:
            chunk_filename = f"chunk_{chunk_index:03d}_{job_id}.wav"
        else:
            chunk_filename = f"chunk_{chunk_index:03d}_{original_filename}.wav"
        chunk_filepath = os.path.join(chunks_dir, chunk_filename)
        
        try:
            # Write audio file
            sf.write(chunk_filepath, chunk_audio, sampling_rate)
            created_files.append(chunk_filepath)
            
            # Log chunk file info
            chunk_duration = len(chunk_audio) / sampling_rate
            file_size_mb = os.path.getsize(chunk_filepath) / (1024 * 1024)
            
            logger.info(f"  ✓ Created: {chunk_filename}")
            logger.debug(f"    - Duration: {chunk_duration/60:.1f}min")
            logger.debug(f"    - File size: {file_size_mb:.1f}MB")
            logger.debug(f"    - Interventions: {len(chunk['interventions'])}")
            logger.debug(f"    - Segments: {len(chunk['segments'])}")
            
        except Exception as e:
            logger.error(f"Failed to create chunk file {chunk_filename}: {e}")
            continue
    
    logger.info(f"Successfully created {len(created_files)} chunk audio files in: {chunks_dir}")
    return created_files


def cleanup_chunk_files(chunk_file_paths: List[str]) -> None:
    """
    Deletes chunk audio files from the filesystem after processing.
    
    Args:
        chunk_file_paths: List of chunk file paths to delete
    """
    if not chunk_file_paths:
        logger.debug("No chunk files to clean up")
        return
    
    logger.info(f"Cleaning up {len(chunk_file_paths)} chunk files...")
    
    deleted_count = 0
    for chunk_path in chunk_file_paths:
        try:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
                deleted_count += 1
                logger.debug(f"Deleted chunk file: {chunk_path}")
            else:
                logger.warning(f"Chunk file not found for deletion: {chunk_path}")
        except Exception as e:
            logger.error(f"Failed to delete chunk file {chunk_path}: {e}")
    
    logger.info(f"Successfully cleaned up {deleted_count}/{len(chunk_file_paths)} chunk files")


def detect_silence_intervals(audio_waveform: np.ndarray, sampling_rate: int) -> List[Tuple[float, float]]:
    """
    Detects silence intervals in audio for natural pause detection.
    
    Args:
        audio_waveform: Audio waveform data
        sampling_rate: Audio sampling rate
        
    Returns:
        List of (start_time, end_time) tuples for silence intervals
    """
    logger.debug(f"Detecting silence intervals in {len(audio_waveform)} samples")
    
    # Convert to dB scale
    audio_db = librosa.amplitude_to_db(np.abs(audio_waveform), ref=np.max)
    
    # Find silence frames
    hop_length = 512
    frame_length = 2048
    silence_frames = audio_db < SILENCE_THRESHOLD
    
    # Convert frames to time
    frame_times = librosa.frames_to_time(np.arange(len(silence_frames)), sr=sampling_rate, hop_length=hop_length)
    
    # Find continuous silence intervals
    silence_intervals = []
    silence_start = None
    
    for i, (is_silent_frame, time) in enumerate(zip(silence_frames, frame_times)):
        if is_silent_frame and silence_start is None:
            silence_start = time
        elif not is_silent_frame and silence_start is not None:
            silence_duration = time - silence_start
            if silence_duration >= MIN_SILENCE_DURATION:
                silence_intervals.append((silence_start, time))
            silence_start = None
    
    # Handle case where audio ends in silence
    if silence_start is not None:
        final_time = len(audio_waveform) / sampling_rate
        if final_time - silence_start >= MIN_SILENCE_DURATION:
            silence_intervals.append((silence_start, final_time))
    
    logger.debug(f"Found {len(silence_intervals)} silence intervals >= {MIN_SILENCE_DURATION}s")
    return silence_intervals


def find_optimal_chunk_points(segment_duration: float, silence_intervals: List[Tuple[float, float]], 
                             segment_start: float) -> List[float]:
    """
    Finds optimal points to split a long segment based on natural pauses.
    
    Args:
        segment_duration: Duration of the segment in seconds
        silence_intervals: List of silence intervals in the entire audio
        segment_start: Start time of the segment in the audio
        
    Returns:
        List of split points (relative to segment start)
    """
    logger.debug(f"Finding chunk points for {segment_duration:.2f}s segment starting at {segment_start:.2f}s")
    
    # Filter silence intervals that fall within this segment
    segment_end = segment_start + segment_duration
    relevant_silences = []
    
    for silence_start, silence_end in silence_intervals:
        # Check if silence overlaps with our segment
        if silence_end > segment_start and silence_start < segment_end:
            # Calculate relative positions within the segment
            rel_start = max(0, silence_start - segment_start)
            rel_end = min(segment_duration, silence_end - segment_start)
            
            if rel_end > rel_start:  # Valid overlap
                relevant_silences.append((rel_start, rel_end))
    
    logger.debug(f"Found {len(relevant_silences)} relevant silence intervals in segment")
    
    # Calculate ideal chunk size and number of chunks
    target_chunk_size = MAX_SEGMENT_DURATION * 0.8  # Aim for 80% of max to have some buffer
    estimated_chunks = max(2, int(np.ceil(segment_duration / target_chunk_size)))
    ideal_chunk_duration = segment_duration / estimated_chunks
    
    logger.debug(f"Target: {estimated_chunks} chunks of ~{ideal_chunk_duration:.2f}s each")
    
    # Find split points
    split_points = []
    current_pos = 0
    
    for chunk_idx in range(1, estimated_chunks):
        # Ideal split position
        ideal_pos = chunk_idx * ideal_chunk_duration
        
        # Find the best silence interval near the ideal position
        best_split = ideal_pos
        min_distance = float('inf')
        
        for silence_start, silence_end in relevant_silences:
            # Consider the middle of each silence interval as potential split point
            silence_mid = (silence_start + silence_end) / 2
            silence_duration = silence_end - silence_start
            
            # Check if this silence is in a reasonable range from ideal position
            distance = abs(silence_mid - ideal_pos)
            
            # Prefer longer silences and those closer to ideal position
            chunk_size_if_split_here = silence_mid - current_pos
            next_chunk_size = segment_duration - silence_mid
            
            # Enhanced constraints - prefer longer silences
            silence_quality = min(silence_duration / MIN_SILENCE_DURATION, 2.0)  # Up to 2x bonus for long silences
            adjusted_distance = distance / silence_quality  # Prefer longer silences
            
            # Check constraints
            if (adjusted_distance < min_distance and 
                chunk_size_if_split_here >= MIN_CHUNK_DURATION and
                next_chunk_size >= MIN_CHUNK_DURATION and
                distance < ideal_chunk_duration * 0.5 and  # Slightly more flexible
                silence_duration >= MIN_SILENCE_DURATION * 0.8):  # Ensure minimum silence quality
                
                min_distance = adjusted_distance
                best_split = silence_mid
        
        # Use the natural silence point directly - overlap will be added when creating chunks
        split_points.append(best_split)
        current_pos = best_split
        
        silence_used = best_split != ideal_pos
        logger.debug(f"      ◦ Chunk {chunk_idx} split at {best_split:.2f}s (ideal: {ideal_pos:.2f}s, natural_silence: {silence_used})")
    
    logger.debug(f"    ◦ Final split points: {split_points}")
    return split_points


def chunk_long_segment(audio_waveform: np.ndarray, sampling_rate: int, 
                      diarization_segment: Dict[str, Any], 
                      audio_silence_intervals: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """
    Chunks a long diarization segment into smaller pieces at natural pauses.
    
    Args:
        audio_waveform: Full audio waveform
        sampling_rate: Audio sampling rate
        diarization_segment: Original segment data with start, end, speaker
        audio_silence_intervals: Pre-computed silence intervals for the entire audio
        
    Returns:
        List of chunked segment dictionaries
    """
    start_time = diarization_segment['start']
    end_time = diarization_segment['end']
    speaker = diarization_segment['speaker']
    duration = end_time - start_time
    
    logger.info(f"  - Chunking long segment: {speaker} ({start_time:.2f}s-{end_time:.2f}s, {duration:.2f}s)")
    
    # Find optimal split points
    split_points = find_optimal_chunk_points(duration, audio_silence_intervals, start_time)
    
    if not split_points:
        logger.warning(f"    • No good split points found for {duration:.2f}s segment, using time-based splits")
        # Fallback to simple time-based splitting
        num_chunks = int(np.ceil(duration / MAX_SEGMENT_DURATION))
        chunk_duration = duration / num_chunks
        split_points = [i * chunk_duration for i in range(1, num_chunks)]
    
    # Create chunks with proper overlap to prevent word loss
    chunks = []
    chunk_boundaries = [0] + split_points + [duration]
    
    for i in range(len(chunk_boundaries) - 1):
        chunk_start = chunk_boundaries[i]
        chunk_end = chunk_boundaries[i + 1]
        
        # Add overlap to preserve context, especially at the beginning of chunks
        overlap_start = chunk_start
        overlap_end = chunk_end
        
        # Add generous overlaps to all chunks to preserve word context
        
        # For chunks after the first one, extend the start backwards to include overlap
        if i > 0:
            # Use minimal overlap to prevent text duplication
            overlap_amount = CHUNK_OVERLAP * 0.3  # 0.3 seconds back overlap to prevent duplication
            overlap_start = max(0, chunk_start - overlap_amount)
            actual_overlap = chunk_start - overlap_start
            logger.debug(f"    • Chunk {i}: Adding {actual_overlap:.3f}s overlap at start")
        
        # For all chunks except the last, extend the end forward to include overlap  
        if i < len(chunk_boundaries) - 1:
            # Add minimal forward overlap to prevent text duplication
            forward_overlap = CHUNK_OVERLAP * 0.3  # 0.3 seconds forward overlap
            overlap_end = min(duration, chunk_end + forward_overlap)
            actual_end_overlap = overlap_end - chunk_end
            logger.debug(f"    • Chunk {i}: Adding {actual_end_overlap:.3f}s overlap at end")
        
        # Calculate absolute times
        abs_start = start_time + overlap_start
        abs_end = start_time + overlap_end
        chunk_duration = abs_end - abs_start
        
        # Extract chunk audio for validation
        start_sample = int(abs_start * sampling_rate)
        end_sample = int(abs_end * sampling_rate)
        chunk_audio = audio_waveform[start_sample:end_sample]
        
        # Validate chunk (skip if too quiet or too short)
        if len(chunk_audio) < MIN_SEGMENT_LENGTH:
            logger.debug(f"    • Skipping chunk {i}: too short ({len(chunk_audio)} samples)")
            continue
            
        # Check if chunk is not silent
        rms_level = np.sqrt(np.mean(chunk_audio**2))
        if rms_level < 0.001:  # Very quiet
            logger.debug(f"    • Skipping chunk {i}: too quiet (RMS: {rms_level:.6f})")
            continue
        
        chunk_data = {
            'start': abs_start,
            'end': abs_end,
            'speaker': speaker,
            'chunk_index': i,
            'original_segment': True,  # Mark as part of original segment
            'chunk_duration': chunk_duration,
            'original_start': start_time + chunk_start,  # Original boundary without overlap
            'original_end': start_time + chunk_end,      # Original boundary without overlap
            'has_start_overlap': i > 0,
            'has_end_overlap': i < len(chunk_boundaries) - 1
        }
        
        chunks.append(chunk_data)
        back_overlap = chunk_start - overlap_start if i > 0 else 0
        forward_overlap = overlap_end - chunk_end if i < len(chunk_boundaries) - 1 else 0
        
        logger.info(f"    • Created chunk {i}: {abs_start:.2f}s-{abs_end:.2f}s ({chunk_duration:.2f}s)")
        logger.info(f"      └─ Original: {start_time + chunk_start:.2f}s-{start_time + chunk_end:.2f}s")
        logger.info(f"      └─ Overlaps: -{back_overlap:.2f}s start, +{forward_overlap:.2f}s end")
    
    # Validate that overlaps are working correctly
    if len(chunks) > 1:
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i-1]
            curr_chunk = chunks[i]
            
            # Check if there's actual overlap in the audio content
            prev_end = prev_chunk['end']
            curr_start = curr_chunk['start']
            
            if curr_start >= prev_end:
                logger.warning(f"    ⚠️  No overlap between chunks {i-1} and {i}: gap of {curr_start - prev_end:.3f}s")
            else:
                overlap_duration = prev_end - curr_start
                logger.info(f"    ✓ Chunks {i-1} and {i} have {overlap_duration:.3f}s overlap")
    
    logger.info(f"    • Successfully chunked {duration:.2f}s segment into {len(chunks)} pieces")
    return chunks


def clean_overlapping_diarization_segments(diarization_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cleans overlapping diarization segments by resolving conflicts.
    
    Algorithm:
    - Detects segments that overlap in time
    - Resolves conflicts by keeping the segment with longer duration
    - Adjusts boundaries to eliminate overlaps
    
    Args:
        diarization_segments: List of raw diarization segments
        
    Returns:
        List of cleaned segments without overlaps
    """
    if not diarization_segments:
        return []
    
    logger.info(f"Cleaning overlapping segments from {len(diarization_segments)} raw segments")
    
    # Sort segments by start time
    segments = sorted(diarization_segments, key=lambda x: x['start'])
    cleaned_segments = []
    conflicts_resolved = 0
    
    i = 0
    while i < len(segments):
        current_segment = segments[i].copy()
        
        # Look for overlaps with subsequent segments
        j = i + 1
        while j < len(segments):
            next_segment = segments[j]
            
            # Check if segments overlap
            if current_segment['end'] > next_segment['start']:
                overlap_duration = current_segment['end'] - next_segment['start']
                current_duration = current_segment['end'] - current_segment['start']
                next_duration = next_segment['end'] - next_segment['start']
                
                logger.debug(f"  - Overlap detected: {current_segment['speaker']} vs {next_segment['speaker']} "
                           f"({overlap_duration:.2f}s overlap)")
                
                # Resolution strategy: keep the longer segment, adjust the shorter one
                if current_duration >= next_duration:
                    # Current segment is longer, adjust next segment's start
                    segments[j] = {
                        'start': current_segment['end'],
                        'end': next_segment['end'],
                        'speaker': next_segment['speaker']
                    }
                    # Skip if the adjusted segment becomes too short
                    if segments[j]['end'] - segments[j]['start'] < MIN_SPEAKER_SEGMENT_DURATION:
                        segments.pop(j)
                        j -= 1
                else:
                    # Next segment is longer, adjust current segment's end
                    current_segment['end'] = next_segment['start']
                    # Skip if the adjusted segment becomes too short
                    if current_segment['end'] - current_segment['start'] < MIN_SPEAKER_SEGMENT_DURATION:
                        break  # Skip adding current segment
                
                conflicts_resolved += 1
                j += 1
            else:
                # No more overlaps
                break
        
        # Add current segment if it's still valid
        if current_segment['end'] - current_segment['start'] >= MIN_SPEAKER_SEGMENT_DURATION:
            cleaned_segments.append(current_segment)
        
        i += 1
    
    logger.info(f"  - Overlap cleanup completed: {conflicts_resolved} conflicts resolved")
    logger.info(f"  - Result: {len(cleaned_segments)} clean segments (removed {len(diarization_segments) - len(cleaned_segments)})")
    
    return cleaned_segments


def remove_consecutive_duplicates(text):
    """
    Remove consecutive duplicate words from text, handling punctuation.
    Example: "los valores y principios principios que nos unen" -> "los valores y principios que nos unen"
    """
    import re
    
    words = text.split()
    if len(words) <= 1:
        return text
    
    cleaned_words = [words[0]]
    
    for i in range(1, len(words)):
        # Normalize for comparison (remove punctuation from end)
        current_word_clean = re.sub(r'[.,;:!?]+$', '', words[i].lower())
        prev_word_clean = re.sub(r'[.,;:!?]+$', '', words[i-1].lower())
        
        if current_word_clean != prev_word_clean or not current_word_clean:
            cleaned_words.append(words[i])
        else:
            logger.debug(f"Removed duplicate word: '{words[i]}'")
    
    return ' '.join(cleaned_words)


def remove_text_duplication(text: str) -> str:
    """
    Clean transcription text by removing multiple spaces and consecutive duplicate words.
    
    Args:
        text: Raw transcription text that may contain duplications from overlapping chunks
        
    Returns:
        Cleaned text with consecutive duplicates removed
    """
    if not text:
        return text
    
    # Clean multiple spaces
    text = ' '.join(text.split())
    
    # Remove consecutive duplicates
    cleaned_text = remove_consecutive_duplicates(text)
    
    # Log if significant duplication was removed
    if len(cleaned_text) < len(text) * 0.9:
        original_words = len(text.split())
        cleaned_words_count = len(cleaned_text.split())
        logger.info(f"Removed consecutive word duplicates: {original_words} → {cleaned_words_count} words")
        logger.debug(f"Original: '{text}'")
        logger.debug(f"Cleaned: '{cleaned_text}'")
    
    return cleaned_text
