"""
Abstract storage interface and base implementations.
"""

from abc import ABC, abstractmethod
from typing import BinaryIO, Optional, Dict, Any, List
from pathlib import Path
import uuid

from ...core.exceptions import StorageError


class StorageBackend(ABC):
    """Abstract storage backend interface."""
    
    @abstractmethod
    async def save_file(
        self, 
        file_content: BinaryIO, 
        filename: str, 
        folder: Optional[str] = None
    ) -> str:
        """
        Save file content to storage.
        
        Args:
            file_content: File content as binary stream
            filename: Original filename
            folder: Optional folder/prefix
            
        Returns:
            Storage path/key for the saved file
        """
        pass
    
    @abstractmethod
    async def get_file(self, file_path: str) -> BinaryIO:
        """
        Retrieve file content from storage.
        
        Args:
            file_path: Storage path/key
            
        Returns:
            File content as binary stream
        """
        pass
    
    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete file from storage.
        
        Args:
            file_path: Storage path/key
            
        Returns:
            True if deleted successfully
        """
        pass
    
    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists in storage.
        
        Args:
            file_path: Storage path/key
            
        Returns:
            True if file exists
        """
        pass
    
    @abstractmethod
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get file metadata.
        
        Args:
            file_path: Storage path/key
            
        Returns:
            File metadata dictionary
        """
        pass
    
    @abstractmethod
    async def list_files(
        self, 
        folder: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> List[str]:
        """
        List files in storage.
        
        Args:
            folder: Optional folder/prefix to filter
            limit: Optional limit on number of files
            
        Returns:
            List of file paths/keys
        """
        pass


class FileManager:
    """
    High-level file management interface.
    Handles file operations with validation and error handling.
    """
    
    def __init__(self, storage_backend: StorageBackend):
        self.storage = storage_backend
        self.allowed_extensions = {
            'audio': {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm', '.aac'},
            'text': {'.txt', '.json', '.csv'},
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp'},
        }
        self.max_file_size = 500 * 1024 * 1024  # 500MB
    
    def validate_file(
        self, 
        filename: str, 
        file_size: int, 
        file_type: str = 'audio'
    ) -> None:
        """
        Validate file before saving.
        
        Args:
            filename: Original filename
            file_size: File size in bytes
            file_type: Type of file (audio, text, image)
            
        Raises:
            StorageError: If validation fails
        """
        # Check file size
        if file_size > self.max_file_size:
            raise StorageError(
                f"File size {file_size} exceeds maximum allowed size {self.max_file_size}"
            )
        
        # Check file extension
        file_ext = Path(filename).suffix.lower()
        allowed_exts = self.allowed_extensions.get(file_type, set())
        
        if file_ext not in allowed_exts:
            raise StorageError(
                f"File extension '{file_ext}' not allowed for type '{file_type}'. "
                f"Allowed extensions: {', '.join(allowed_exts)}"
            )
    
    def generate_unique_filename(self, original_filename: str) -> str:
        """
        Generate unique filename to avoid conflicts.
        
        Args:
            original_filename: Original filename
            
        Returns:
            Unique filename with UUID prefix
        """
        file_path = Path(original_filename)
        unique_id = str(uuid.uuid4())
        
        return f"{unique_id}_{file_path.stem}{file_path.suffix}"
    
    async def save_audio_file(
        self, 
        file_content: BinaryIO, 
        original_filename: str,
        job_id: Optional[str] = None
    ) -> str:
        """
        Save audio file with validation.
        
        Args:
            file_content: Audio file content
            original_filename: Original filename
            job_id: Optional job ID for organization
            
        Returns:
            Storage path for saved file
        """
        # Get file size
        file_content.seek(0, 2)  # Seek to end
        file_size = file_content.tell()
        file_content.seek(0)  # Reset to beginning
        
        # Validate file
        self.validate_file(original_filename, file_size, 'audio')
        
        # Generate unique filename
        unique_filename = self.generate_unique_filename(original_filename)
        
        # Determine folder structure
        folder = "audio_files"
        if job_id:
            folder = f"audio_files/{job_id}"
        
        # Save file
        file_path = await self.storage.save_file(
            file_content, 
            unique_filename, 
            folder
        )
        
        return file_path
    
    async def save_processed_audio(
        self, 
        file_content: BinaryIO, 
        filename: str,
        job_id: str,
        processing_stage: str = "processed"
    ) -> str:
        """
        Save processed audio file (converted, chunked, etc.).
        
        Args:
            file_content: Processed audio content
            filename: Filename for processed audio
            job_id: Job ID
            processing_stage: Stage of processing (converted, chunked, etc.)
            
        Returns:
            Storage path for saved file
        """
        folder = f"audio_files/{job_id}/{processing_stage}"
        
        file_path = await self.storage.save_file(
            file_content,
            filename,
            folder
        )
        
        return file_path
    
    async def cleanup_job_files(self, job_id: str) -> Dict[str, Any]:
        """
        Clean up all files associated with a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Cleanup summary
        """
        files_deleted = 0
        space_freed = 0
        errors = []
        
        try:
            # List all files for the job
            job_files = await self.storage.list_files(f"audio_files/{job_id}")
            
            for file_path in job_files:
                try:
                    # Get file info before deletion
                    file_info = await self.storage.get_file_info(file_path)
                    file_size = file_info.get('size', 0)
                    
                    # Delete file
                    if await self.storage.delete_file(file_path):
                        files_deleted += 1
                        space_freed += file_size
                    
                except Exception as e:
                    errors.append(f"Failed to delete {file_path}: {str(e)}")
        
        except Exception as e:
            errors.append(f"Failed to list job files: {str(e)}")
        
        return {
            "files_deleted": files_deleted,
            "space_freed_bytes": space_freed,
            "errors": errors
        }