"""
Local filesystem storage implementation.
"""

import os
import shutil
import aiofiles
from typing import BinaryIO, Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime

from .base import StorageBackend
from ...core.exceptions import StorageError
from ...core.logging import get_logger

logger = get_logger(__name__)


class LocalFileStorage(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized local storage at: {self.base_path}")
    
    def _get_full_path(self, file_path: str) -> Path:
        """Get full filesystem path from storage path."""
        return self.base_path / file_path
    
    def _get_storage_path(self, filename: str, folder: Optional[str] = None) -> str:
        """Generate storage path from filename and folder."""
        if folder:
            return f"{folder}/{filename}"
        return filename
    
    async def save_file(
        self, 
        file_content: BinaryIO, 
        filename: str, 
        folder: Optional[str] = None
    ) -> str:
        """Save file to local filesystem."""
        try:
            storage_path = self._get_storage_path(filename, folder)
            full_path = self._get_full_path(storage_path)
            
            # Create directory if it doesn't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save file
            async with aiofiles.open(full_path, 'wb') as f:
                # Reset file pointer to beginning
                file_content.seek(0)
                content = file_content.read()
                await f.write(content)
            
            logger.debug(f"Saved file to: {full_path}")
            return storage_path
            
        except Exception as e:
            logger.error(f"Failed to save file {filename}: {e}")
            raise StorageError(f"Failed to save file: {e}")
    
    async def get_file(self, file_path: str) -> BinaryIO:
        """Retrieve file from local filesystem."""
        try:
            full_path = self._get_full_path(file_path)
            
            if not full_path.exists():
                raise StorageError(f"File not found: {file_path}")
            
            # Return file handle
            return open(full_path, 'rb')
            
        except Exception as e:
            logger.error(f"Failed to get file {file_path}: {e}")
            raise StorageError(f"Failed to retrieve file: {e}")
    
    async def delete_file(self, file_path: str) -> bool:
        """Delete file from local filesystem."""
        try:
            full_path = self._get_full_path(file_path)
            
            if not full_path.exists():
                logger.warning(f"File not found for deletion: {file_path}")
                return False
            
            full_path.unlink()
            logger.debug(f"Deleted file: {full_path}")
            
            # Clean up empty directories
            try:
                full_path.parent.rmdir()
            except OSError:
                # Directory not empty, which is fine
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")
            raise StorageError(f"Failed to delete file: {e}")
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in local filesystem."""
        full_path = self._get_full_path(file_path)
        return full_path.exists()
    
    async def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file metadata from local filesystem."""
        try:
            full_path = self._get_full_path(file_path)
            
            if not full_path.exists():
                raise StorageError(f"File not found: {file_path}")
            
            stat = full_path.stat()
            
            return {
                "path": file_path,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "is_file": full_path.is_file(),
                "extension": full_path.suffix.lower()
            }
            
        except Exception as e:
            logger.error(f"Failed to get file info for {file_path}: {e}")
            raise StorageError(f"Failed to get file info: {e}")
    
    async def list_files(
        self, 
        folder: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> List[str]:
        """List files in local filesystem."""
        try:
            if folder:
                search_path = self._get_full_path(folder)
            else:
                search_path = self.base_path
            
            if not search_path.exists():
                return []
            
            files = []
            for item in search_path.rglob('*'):
                if item.is_file():
                    # Get relative path from base_path
                    relative_path = item.relative_to(self.base_path)
                    files.append(str(relative_path))
                    
                    if limit and len(files) >= limit:
                        break
            
            return sorted(files)
            
        except Exception as e:
            logger.error(f"Failed to list files in {folder}: {e}")
            raise StorageError(f"Failed to list files: {e}")
    
    async def get_directory_size(self, folder: Optional[str] = None) -> int:
        """Get total size of directory in bytes."""
        try:
            if folder:
                search_path = self._get_full_path(folder)
            else:
                search_path = self.base_path
            
            if not search_path.exists():
                return 0
            
            total_size = 0
            for item in search_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
            
            return total_size
            
        except Exception as e:
            logger.error(f"Failed to calculate directory size: {e}")
            return 0
    
    async def cleanup_empty_directories(self, folder: Optional[str] = None) -> int:
        """Remove empty directories recursively."""
        try:
            if folder:
                search_path = self._get_full_path(folder)
            else:
                search_path = self.base_path
            
            if not search_path.exists():
                return 0
            
            removed_count = 0
            
            # Walk directories bottom-up
            for item in search_path.rglob('*'):
                if item.is_dir() and item != search_path:
                    try:
                        item.rmdir()  # Only removes if empty
                        removed_count += 1
                        logger.debug(f"Removed empty directory: {item}")
                    except OSError:
                        # Directory not empty, continue
                        pass
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup empty directories: {e}")
            return 0