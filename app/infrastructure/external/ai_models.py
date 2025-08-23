"""
AI model management and loading infrastructure.
Handles Whisper and diarization model lifecycle with caching and optimization.
"""

import time
import threading
from typing import Optional, Dict, Any, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from contextlib import contextmanager

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from pyannote.audio import Pipeline

from ...core.config import settings
from ...core.exceptions import ModelLoadingError
from ...core.logging import get_logger
from ...domain.shared.enums import ModelType, DeviceType
from ...domain.shared.value_objects import ModelConfiguration

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_type: ModelType
    model_path: str
    device: str
    loaded_at: float
    memory_usage_mb: Optional[float] = None
    parameters_count: Optional[int] = None


class ModelManager(ABC):
    """Abstract base class for model managers."""
    
    @abstractmethod
    def load_model(self, config: ModelConfiguration) -> Any:
        """Load a model with given configuration."""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get information about the loaded model."""
        pass


class WhisperModelManager(ModelManager):
    """Manages Whisper model loading and lifecycle."""
    
    def __init__(self):
        self.processor: Optional[WhisperProcessor] = None
        self.model: Optional[WhisperForConditionalGeneration] = None
        self.config: Optional[ModelConfiguration] = None
        self.model_info: Optional[ModelInfo] = None
        self._lock = threading.Lock()
    
    def load_model(self, config: ModelConfiguration) -> Tuple[WhisperProcessor, WhisperForConditionalGeneration]:
        """
        Load Whisper model with given configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Tuple of (processor, model)
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        with self._lock:
            logger.info(f"Loading Whisper model: {config.model_path}")
            start_time = time.time()
            
            try:
                # Validate device
                device = self._validate_device(config.device)
                
                # Load processor
                logger.info("Loading Whisper processor...")
                processor = WhisperProcessor.from_pretrained(config.model_path)
                
                # Load model
                logger.info("Loading Whisper model...")
                model = WhisperForConditionalGeneration.from_pretrained(
                    config.model_path,
                    torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                ).to(device)
                
                # Configure generation settings
                model.generation_config.task = "transcribe"
                model.generation_config.forced_decoder_ids = None
                
                # Apply additional parameters
                if config.additional_params:
                    for key, value in config.additional_params.items():
                        if hasattr(model.generation_config, key):
                            setattr(model.generation_config, key, value)
                
                # Calculate model info
                parameters_count = sum(p.numel() for p in model.parameters())
                memory_usage = self._estimate_memory_usage(model, device)
                
                loading_time = time.time() - start_time
                
                # Store model info
                self.model_info = ModelInfo(
                    model_type=ModelType.WHISPER,
                    model_path=config.model_path,
                    device=device,
                    loaded_at=time.time(),
                    memory_usage_mb=memory_usage,
                    parameters_count=parameters_count
                )
                
                self.processor = processor
                self.model = model
                self.config = config
                
                logger.info(f"Whisper model loaded successfully in {loading_time:.2f}s")
                logger.info(f"Model parameters: {parameters_count:,}")
                logger.info(f"Estimated memory usage: {memory_usage:.1f}MB")
                
                return processor, model
                
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}", exc_info=True)
                raise ModelLoadingError(f"Whisper model loading failed: {e}")
    
    def unload_model(self) -> None:
        """Unload Whisper model to free memory."""
        with self._lock:
            if self.model is not None:
                logger.info("Unloading Whisper model")
                
                # Move model to CPU and delete
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                del self.model
                del self.processor
                
                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.model = None
                self.processor = None
                self.config = None
                self.model_info = None
                
                logger.info("Whisper model unloaded successfully")
    
    def is_loaded(self) -> bool:
        """Check if Whisper model is loaded."""
        return self.model is not None and self.processor is not None
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get information about the loaded Whisper model."""
        return self.model_info
    
    def _validate_device(self, device: str) -> str:
        """Validate and adjust device specification."""
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(f"CUDA device {device} requested but not available, using CPU")
            return "cpu"
        
        return device
    
    def _estimate_memory_usage(self, model: torch.nn.Module, device: str) -> float:
        """Estimate model memory usage in MB."""
        try:
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
            
            # Add some overhead for activations and computation
            overhead_factor = 1.5 if device.startswith("cuda") else 1.2
            total_bytes = (param_size + buffer_size) * overhead_factor
            
            return total_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            return 0.0


class DiarizationModelManager(ModelManager):
    """Manages diarization model loading and lifecycle."""
    
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.config: Optional[ModelConfiguration] = None
        self.model_info: Optional[ModelInfo] = None
        self._lock = threading.Lock()
    
    def load_model(self, config: ModelConfiguration) -> Pipeline:
        """
        Load diarization model with given configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Diarization pipeline
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        with self._lock:
            logger.info(f"Loading diarization model: {config.model_path}")
            start_time = time.time()
            
            try:
                # Get HuggingFace token
                hf_token = settings.ai_models.hf_token
                if not hf_token:
                    logger.warning("HF_TOKEN not set, diarization may fail for gated models")
                
                # Validate device
                device = self._validate_device(config.device)
                
                # Load pipeline
                pipeline = Pipeline.from_pretrained(
                    config.model_path,
                    use_auth_token=hf_token
                )
                
                # Move to device
                pipeline.to(torch.device(device))
                
                # Configure pipeline parameters
                if hasattr(pipeline, 'segmentation'):
                    pipeline.segmentation.onset = settings.diarization.onset
                    pipeline.segmentation.offset = settings.diarization.offset
                    pipeline.segmentation.min_duration_on = settings.diarization.min_duration_on
                    pipeline.segmentation.min_duration_off = settings.diarization.min_duration_off
                
                # Apply additional parameters
                if config.additional_params:
                    for key, value in config.additional_params.items():
                        if hasattr(pipeline, key):
                            setattr(pipeline, key, value)
                
                loading_time = time.time() - start_time
                
                # Store model info
                self.model_info = ModelInfo(
                    model_type=ModelType.DIARIZATION,
                    model_path=config.model_path,
                    device=device,
                    loaded_at=time.time(),
                    memory_usage_mb=None,  # Difficult to estimate for pipeline
                    parameters_count=None
                )
                
                self.pipeline = pipeline
                self.config = config
                
                logger.info(f"Diarization model loaded successfully in {loading_time:.2f}s")
                
                return pipeline
                
            except Exception as e:
                logger.error(f"Failed to load diarization model: {e}", exc_info=True)
                raise ModelLoadingError(f"Diarization model loading failed: {e}")
    
    def unload_model(self) -> None:
        """Unload diarization model to free memory."""
        with self._lock:
            if self.pipeline is not None:
                logger.info("Unloading diarization model")
                
                # Clear pipeline
                del self.pipeline
                
                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.pipeline = None
                self.config = None
                self.model_info = None
                
                logger.info("Diarization model unloaded successfully")
    
    def is_loaded(self) -> bool:
        """Check if diarization model is loaded."""
        return self.pipeline is not None
    
    def get_model_info(self) -> Optional[ModelInfo]:
        """Get information about the loaded diarization model."""
        return self.model_info
    
    def _validate_device(self, device: str) -> str:
        """Validate and adjust device specification."""
        if device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(f"CUDA device {device} requested but not available, using CPU")
            return "cpu"
        
        return device


class AIModelRegistry:
    """
    Central registry for managing multiple AI models.
    Provides model loading, caching, and lifecycle management.
    """
    
    def __init__(self):
        self.whisper_manager = WhisperModelManager()
        self.diarization_manager = DiarizationModelManager()
        self._initialized = False
        self._lock = threading.Lock()
    
    def initialize_models(self, force_reload: bool = False) -> None:
        """
        Initialize all required models based on configuration.
        
        Args:
            force_reload: Whether to force reload even if already loaded
        """
        with self._lock:
            if self._initialized and not force_reload:
                logger.info("Models already initialized")
                return
            
            logger.info("Initializing AI models...")
            start_time = time.time()
            
            try:
                # Initialize Whisper model
                whisper_config = ModelConfiguration(
                    model_path=settings.ai_models.whisper_model_path,
                    device=settings.ai_models.device,
                    batch_size=settings.ai_models.batch_size,
                    additional_params={
                        "beam_size": settings.ai_models.beam_size,
                        "temperature": settings.ai_models.temperature,
                        "no_repeat_ngram_size": settings.ai_models.no_repeat_ngram_size
                    }
                )
                
                self.whisper_manager.load_model(whisper_config)
                
                # Initialize diarization model
                diarization_config = ModelConfiguration(
                    model_path=settings.ai_models.diarization_model_path,
                    device=settings.ai_models.device,
                    batch_size=1,  # Diarization typically uses batch size 1
                    additional_params={}
                )
                
                self.diarization_manager.load_model(diarization_config)
                
                self._initialized = True
                
                total_time = time.time() - start_time
                logger.info(f"All AI models initialized successfully in {total_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to initialize AI models: {e}", exc_info=True)
                raise ModelLoadingError(f"Model initialization failed: {e}")
    
    def get_whisper_models(self) -> Tuple[WhisperProcessor, WhisperForConditionalGeneration]:
        """
        Get loaded Whisper models.
        
        Returns:
            Tuple of (processor, model)
            
        Raises:
            ModelLoadingError: If models are not loaded
        """
        if not self.whisper_manager.is_loaded():
            raise ModelLoadingError("Whisper models not loaded")
        
        return self.whisper_manager.processor, self.whisper_manager.model
    
    def get_diarization_pipeline(self) -> Pipeline:
        """
        Get loaded diarization pipeline.
        
        Returns:
            Diarization pipeline
            
        Raises:
            ModelLoadingError: If pipeline is not loaded
        """
        if not self.diarization_manager.is_loaded():
            raise ModelLoadingError("Diarization pipeline not loaded")
        
        return self.diarization_manager.pipeline
    
    def get_models_info(self) -> Dict[str, Any]:
        """Get information about all loaded models."""
        return {
            "whisper": self.whisper_manager.get_model_info(),
            "diarization": self.diarization_manager.get_model_info(),
            "total_models": sum(1 for m in [self.whisper_manager, self.diarization_manager] if m.is_loaded()),
            "initialization_status": self._initialized
        }
    
    def unload_all_models(self) -> None:
        """Unload all models to free memory."""
        with self._lock:
            logger.info("Unloading all AI models")
            
            self.whisper_manager.unload_model()
            self.diarization_manager.unload_model()
            
            self._initialized = False
            logger.info("All AI models unloaded")
    
    @contextmanager
    def model_context(self):
        """Context manager for model usage with automatic cleanup."""
        try:
            if not self._initialized:
                self.initialize_models()
            yield self
        except Exception as e:
            logger.error(f"Error in model context: {e}")
            raise
        # Note: We don't automatically unload models here as they might be reused


# Global model registry instance
_model_registry: Optional[AIModelRegistry] = None


def get_model_registry() -> AIModelRegistry:
    """
    Get the global AI model registry instance.
    
    Returns:
        AI model registry
    """
    global _model_registry
    
    if _model_registry is None:
        _model_registry = AIModelRegistry()
    
    return _model_registry


def initialize_ai_models(force_reload: bool = False) -> None:
    """
    Initialize AI models using global registry.
    
    Args:
        force_reload: Whether to force reload even if already loaded
    """
    registry = get_model_registry()
    registry.initialize_models(force_reload=force_reload)