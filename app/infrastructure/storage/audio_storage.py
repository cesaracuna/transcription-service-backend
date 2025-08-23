"""
Specialized audio file storage with format conversion and optimization.
"""

import subprocess
import tempfile
import os
from typing import BinaryIO, Optional, Dict, Any, Tuple
from pathlib import Path
import asyncio

from .base import FileManager, StorageBackend
from ...core.config import settings
from ...core.exceptions import StorageError, AudioProcessingError
from ...core.logging import get_logger
from ...domain.shared.value_objects import AudioMetadata

logger = get_logger(__name__)


class AudioFileManager(FileManager):
    """
    Specialized file manager for audio files with format conversion and optimization.
    """
    
    def __init__(self, storage_backend: StorageBackend):
        super().__init__(storage_backend)
        self.target_sample_rate = settings.audio.sample_rate
        self.target_format = "wav"
        
    async def save_and_convert_audio(
        self, 
        file_content: BinaryIO, 
        original_filename: str,
        job_id: str,
        convert_to_standard: bool = True
    ) -> Tuple[str, AudioMetadata]:
        """
        Save audio file and optionally convert to standard format.
        
        Args:
            file_content: Audio file content
            original_filename: Original filename
            job_id: Job ID for organization
            convert_to_standard: Whether to convert to standard format
            
        Returns:
            Tuple of (storage_path, audio_metadata)
        """
        logger.info(f"Processing audio file: {original_filename}")
        
        try:
            # Save original file first
            original_path = await self.save_audio_file(
                file_content, 
                original_filename, 
                job_id
            )
            
            # Get audio metadata
            metadata = await self._get_audio_metadata(original_path)
            
            if convert_to_standard:
                # Convert to standard format
                converted_path = await self._convert_audio_format(
                    original_path, 
                    job_id
                )
                
                # Update metadata for converted file
                converted_metadata = await self._get_audio_metadata(converted_path)
                
                return converted_path, converted_metadata
            else:
                return original_path, metadata
                
        except Exception as e:
            logger.error(f"Failed to process audio file: {e}")
            raise AudioProcessingError(f"Audio processing failed: {e}")
    
    async def _convert_audio_format(
        self, 
        input_path: str, 
        job_id: str
    ) -> str:
        """
        Convert audio to standard format (16kHz mono WAV).
        
        Args:
            input_path: Input audio file path
            job_id: Job ID
            
        Returns:
            Path to converted audio file
        """
        logger.info(f"Converting audio to standard format: {input_path}")
        
        try:
            # Get full path for input file
            full_input_path = self.storage._get_full_path(input_path)
            
            # Generate output filename
            input_filename = Path(input_path).stem
            output_filename = f"{input_filename}_16khz_mono.wav"
            
            # Create temporary file for conversion
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_output_path = temp_file.name
            
            # FFmpeg command for conversion
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", str(full_input_path),
                "-map", "0:a",              # Map all audio tracks
                "-ac", "1",                 # Mono (1 channel)
                "-ar", str(self.target_sample_rate),  # Sample rate
                "-c:a", "pcm_s16le",        # WAV codec
                "-y",                       # Overwrite output
                temp_output_path
            ]
            
            logger.debug(f"Running FFmpeg: {' '.join(ffmpeg_cmd)}")
            
            # Run conversion
            result = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
                logger.error(f"FFmpeg conversion failed: {error_msg}")
                raise AudioProcessingError(f"Audio conversion failed: {error_msg}")
            
            # Save converted file
            with open(temp_output_path, 'rb') as converted_file:
                converted_path = await self.save_processed_audio(
                    converted_file,
                    output_filename,
                    job_id,
                    "converted"
                )
            
            # Clean up temporary file
            os.unlink(temp_output_path)
            
            logger.info(f"Audio conversion completed: {converted_path}")
            return converted_path
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise AudioProcessingError(f"Failed to convert audio: {e}")
    
    async def _get_audio_metadata(self, file_path: str) -> AudioMetadata:
        """
        Extract audio metadata using FFprobe.
        
        Args:
            file_path: Audio file path
            
        Returns:
            Audio metadata
        """
        try:
            full_path = self.storage._get_full_path(file_path)
            
            # FFprobe command to get metadata
            ffprobe_cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(full_path)
            ]
            
            result = await asyncio.create_subprocess_exec(
                *ffprobe_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown FFprobe error"
                logger.error(f"FFprobe failed: {error_msg}")
                # Return default metadata if FFprobe fails
                return self._get_default_metadata(file_path)
            
            # Parse JSON output
            import json
            metadata = json.loads(stdout.decode())
            
            # Extract audio stream info
            audio_stream = None
            for stream in metadata.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                logger.warning("No audio stream found in metadata")
                return self._get_default_metadata(file_path)
            
            # Extract metadata
            format_info = metadata.get('format', {})
            
            duration_seconds = float(format_info.get('duration', 0))
            sample_rate = int(audio_stream.get('sample_rate', self.target_sample_rate))
            channels = int(audio_stream.get('channels', 1))
            file_size_bytes = int(format_info.get('size', 0))
            
            return AudioMetadata(
                duration_seconds=duration_seconds,
                sample_rate=sample_rate,
                channels=channels,
                format=Path(file_path).suffix.lstrip('.').lower(),
                file_size_bytes=file_size_bytes,
                rms_level=0.0,  # Would need additional processing to calculate
                peak_level=0.0  # Would need additional processing to calculate
            )
            
        except Exception as e:
            logger.error(f"Failed to extract audio metadata: {e}")
            return self._get_default_metadata(file_path)
    
    def _get_default_metadata(self, file_path: str) -> AudioMetadata:
        """Get default metadata when extraction fails."""
        return AudioMetadata(
            duration_seconds=0.0,
            sample_rate=self.target_sample_rate,
            channels=1,
            format=Path(file_path).suffix.lstrip('.').lower() or "wav",
            file_size_bytes=0,
            rms_level=0.0,
            peak_level=0.0
        )
    
    async def create_audio_chunks(
        self, 
        audio_path: str, 
        job_id: str,
        chunk_intervals: list[Tuple[float, float]]
    ) -> list[str]:
        """
        Create audio chunks from intervals for processing.
        
        Args:
            audio_path: Source audio file path
            job_id: Job ID
            chunk_intervals: List of (start_time, end_time) tuples
            
        Returns:
            List of chunk file paths
        """
        logger.info(f"Creating {len(chunk_intervals)} audio chunks")
        
        chunk_paths = []
        full_input_path = self.storage._get_full_path(audio_path)
        
        for i, (start_time, end_time) in enumerate(chunk_intervals):
            try:
                duration = end_time - start_time
                chunk_filename = f"chunk_{i:03d}_{start_time:.2f}s_{duration:.2f}s.wav"
                
                # Create temporary file for chunk
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_chunk_path = temp_file.name
                
                # FFmpeg command to extract chunk
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-i", str(full_input_path),
                    "-ss", str(start_time),      # Start time
                    "-t", str(duration),         # Duration
                    "-c", "copy",                # Copy without re-encoding
                    "-y",                        # Overwrite output
                    temp_chunk_path
                ]
                
                # Run extraction
                result = await asyncio.create_subprocess_exec(
                    *ffmpeg_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await result.communicate()
                
                if result.returncode != 0:
                    error_msg = stderr.decode() if stderr else "Unknown FFmpeg error"
                    logger.error(f"Chunk extraction failed: {error_msg}")
                    continue
                
                # Save chunk
                with open(temp_chunk_path, 'rb') as chunk_file:
                    chunk_path = await self.save_processed_audio(
                        chunk_file,
                        chunk_filename,
                        job_id,
                        "chunks"
                    )
                    chunk_paths.append(chunk_path)
                
                # Clean up temporary file
                os.unlink(temp_chunk_path)
                
                logger.debug(f"Created chunk {i}: {chunk_path}")
                
            except Exception as e:
                logger.error(f"Failed to create chunk {i}: {e}")
                continue
        
        logger.info(f"Successfully created {len(chunk_paths)} audio chunks")
        return chunk_paths
    
    async def cleanup_chunks(self, job_id: str) -> Dict[str, Any]:
        """
        Clean up audio chunks for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Cleanup summary
        """
        logger.info(f"Cleaning up audio chunks for job {job_id}")
        
        try:
            chunks_folder = f"audio_files/{job_id}/chunks"
            chunk_files = await self.storage.list_files(chunks_folder)
            
            files_deleted = 0
            space_freed = 0
            
            for chunk_path in chunk_files:
                try:
                    file_info = await self.storage.get_file_info(chunk_path)
                    file_size = file_info.get('size', 0)
                    
                    if await self.storage.delete_file(chunk_path):
                        files_deleted += 1
                        space_freed += file_size
                        
                except Exception as e:
                    logger.error(f"Failed to delete chunk {chunk_path}: {e}")
            
            # Clean up empty chunk directory
            await self.storage.cleanup_empty_directories(chunks_folder)
            
            logger.info(f"Chunk cleanup completed: {files_deleted} files, {space_freed} bytes")
            
            return {
                "files_deleted": files_deleted,
                "space_freed_bytes": space_freed,
                "errors": []
            }
            
        except Exception as e:
            logger.error(f"Chunk cleanup failed: {e}")
            return {
                "files_deleted": 0,
                "space_freed_bytes": 0,
                "errors": [str(e)]
            }