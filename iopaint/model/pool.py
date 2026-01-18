from typing import Dict, Any, Optional, List
import torch
from loguru import logger


class SharedComponentType:
    """Constants for different types of shared components."""
    VAE_SD15 = "vae_sd15"
    VAE_SDXL = "vae_sdxl"
    TEXT_ENCODER_SD15 = "text_encoder_sd15"
    TEXT_ENCODER_SDXL = "text_encoder_sdxl"
    CONTROLNET_SD15 = "controlnet_sd15"
    CONTROLNET_SDXL = "controlnet_sdxl"
    UNET_SD15 = "unet_sd15"  # Generally not shared due to different weights
    UNET_SDXL = "unet_sdxl"  # Generally not shared due to different weights


class ModelPool:
    """Pool of loaded models with component-level weight sharing support.

    This pool manages shared components (VAE, text encoders, etc.) between
    similar model architectures to reduce memory usage and improve loading speed.
    """

    def __init__(self, max_memory_mb: int = 16384):
        self.max_memory_mb = max_memory_mb
        self.current_memory_mb = 0

        # Pool of loaded models: model_name -> model_instance
        self._models: Dict[str, Any] = {}

        # Shared components: component_type -> component_instance
        self._shared_components: Dict[str, Any] = {}

        # Reference counts for shared components
        self._component_refcount: Dict[str, int] = {}

        # Memory usage tracking
        self._component_memory: Dict[str, int] = {}  # component -> memory usage in MB

    def get_or_load_model(self, model_name: str, model_info, device: torch.device, **kwargs) -> Any:
        """Get model from pool or load it if not present."""
        if model_name in self._models:
            logger.debug(f"Returning cached model: {model_name}")
            return self._models[model_name]

        # Load new model
        logger.info(f"Loading new model into pool: {model_name}")
        model = self._load_model_with_sharing(model_name, model_info, device, **kwargs)
        self._models[model_name] = model

        return model

    def _load_model_with_sharing(self, model_name: str, model_info, device: torch.device, **kwargs) -> Any:
        """Load a model with component sharing where possible."""
        from iopaint.model import models

        model_cls = models[model_name]

        # Check if this model supports component sharing
        if hasattr(model_cls, 'get_shared_components'):
            return self._load_model_with_component_sharing(model_cls, model_name, device, **kwargs)
        else:
            # Load normally without sharing
            return model_cls(device, **kwargs)

    def _load_model_with_component_sharing(self, model_cls, model_name: str, device: torch.device, **kwargs) -> Any:
        """Load model using shared components."""
        shared_components = {}

        # Try to get shared components for this model type
        component_types = model_cls.get_shared_components()
        for comp_type in component_types:
            component = self.get_shared_component(comp_type, model_name, device)
            if component is not None:
                shared_components[comp_type] = component

        # Load model with shared components
        kwargs['shared_components'] = shared_components
        model = model_cls(device, **kwargs)

        # Track memory usage
        self._update_memory_usage()

        return model

    def get_shared_component(self, component_type: str, model_name: str, device: torch.device) -> Optional[Any]:
        """Get a shared component, loading it if necessary."""
        if component_type in self._shared_components:
            # Increment reference count
            self._component_refcount[component_type] += 1
            logger.debug(f"Reusing shared component {component_type} (refcount: {self._component_refcount[component_type]})")
            return self._shared_components[component_type]

        # Load new shared component
        component = self._load_shared_component(component_type, model_name, device)
        if component is not None:
            self._shared_components[component_type] = component
            self._component_refcount[component_type] = 1
            # Estimate memory usage (rough approximation)
            self._component_memory[component_type] = self._estimate_component_memory(component)
            self.current_memory_mb += self._component_memory[component_type]
            logger.debug(f"Loaded new shared component {component_type} ({self._component_memory[component_type]}MB)")

        return component

    def _load_shared_component(self, component_type: str, model_name: str, device: torch.device) -> Optional[Any]:
        """Load a specific shared component based on type."""
        try:
            if component_type == "vae_sd15":
                return self._load_sd15_vae(device)
            elif component_type == "text_encoder_sd15":
                return self._load_sd15_text_encoder(device)
            elif component_type == "vae_sdxl":
                return self._load_sdxl_vae(device)
            elif component_type == "text_encoder_sdxl":
                return self._load_sdxl_text_encoder(device)
            else:
                logger.debug(f"Unknown component type: {component_type}")
                return None
        except Exception as e:
            logger.warning(f"Failed to load shared component {component_type}: {e}")
            return None

    def _load_sd15_vae(self, device: torch.device):
        """Load SD 1.5 VAE component."""
        from diffusers import AutoencoderKL
        from .utils import handle_from_pretrained_exceptions

        vae = handle_from_pretrained_exceptions(
            AutoencoderKL.from_pretrained,
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            local_files_only=True,  # Only use cached/local files
        )
        return vae.to(device)

    def _load_sd15_text_encoder(self, device: torch.device):
        """Load SD 1.5 text encoder component."""
        from transformers import CLIPTextModel
        from .utils import handle_from_pretrained_exceptions

        text_encoder = handle_from_pretrained_exceptions(
            CLIPTextModel.from_pretrained,
            pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
            subfolder="text_encoder",
            local_files_only=True,  # Only use cached/local files
        )
        return text_encoder.to(device)

    def _load_sdxl_vae(self, device: torch.device):
        """Load SDXL VAE component."""
        from diffusers import AutoencoderKL
        from .utils import handle_from_pretrained_exceptions

        vae = handle_from_pretrained_exceptions(
            AutoencoderKL.from_pretrained,
            pretrained_model_name_or_path="madebyollin/sdxl-vae-fp16-fix",
            local_files_only=True,  # Only use cached/local files
        )
        return vae.to(device)

    def _load_sdxl_text_encoder(self, device: torch.device):
        """Load SDXL text encoder components (dual encoders)."""
        from transformers import CLIPTextModel, CLIPTextModelWithProjection
        from .utils import handle_from_pretrained_exceptions

        # SDXL has two text encoders
        text_encoder = handle_from_pretrained_exceptions(
            CLIPTextModel.from_pretrained,
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="text_encoder",
            local_files_only=True,
        )

        text_encoder_2 = handle_from_pretrained_exceptions(
            CLIPTextModelWithProjection.from_pretrained,
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            subfolder="text_encoder_2",
            local_files_only=True,
        )

        # Return as a dict since SDXL needs both
        return {
            "text_encoder": text_encoder.to(device),
            "text_encoder_2": text_encoder_2.to(device),
        }

    def release_model(self, model_name: str):
        """Release a model from the pool."""
        if model_name not in self._models:
            return

        model = self._models[model_name]

        # Release shared components used by this model
        if hasattr(model, 'get_used_components'):
            used_components = model.get_used_components()
            for comp_type in used_components:
                self.release_shared_component(comp_type)

        # Remove model from pool
        del self._models[model_name]
        logger.debug(f"Released model from pool: {model_name}")

    def release_shared_component(self, component_type: str):
        """Release reference to a shared component."""
        if component_type not in self._component_refcount:
            return

        self._component_refcount[component_type] -= 1

        if self._component_refcount[component_type] <= 0:
            # No more references, can release
            if component_type in self._shared_components:
                memory_freed = self._component_memory.get(component_type, 0)
                self.current_memory_mb -= memory_freed

                # Clean up component
                component = self._shared_components[component_type]
                if hasattr(component, 'to'):
                    component.to('cpu')  # Move to CPU to free GPU memory
                del self._shared_components[component_type]
                del self._component_refcount[component_type]
                del self._component_memory[component_type]

                # Force garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.debug(f"Released shared component {component_type} ({memory_freed}MB freed)")

    def _estimate_component_memory(self, component) -> int:
        """Estimate memory usage of a component in MB."""
        if hasattr(component, 'parameters'):
            # Rough estimate: 4 bytes per parameter + overhead
            param_count = sum(p.numel() for p in component.parameters())
            return max(1, param_count * 4 // (1024 * 1024))  # Convert to MB
        return 1  # Minimum 1MB estimate

    def _update_memory_usage(self):
        """Update total memory usage tracking."""
        # This is a simplified implementation
        # In practice, you'd want more accurate GPU memory tracking
        pass

    def evict_least_recently_used(self, target_memory_mb: int):
        """Evict models to free up memory."""
        # Simple LRU eviction - in practice, you'd track access times
        models_to_evict = []

        # Calculate how much memory to free
        memory_to_free = self.current_memory_mb - (self.max_memory_mb - target_memory_mb)
        if memory_to_free <= 0:
            return

        freed_memory = 0
        for model_name in list(self._models.keys()):
            if freed_memory >= memory_to_free:
                break
            self.release_model(model_name)
            # Estimate memory freed (simplified)
            freed_memory += 100  # Rough estimate per model

        logger.info(f"Evicted models to free {freed_memory}MB memory")

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        return {
            'total_mb': self.current_memory_mb,
            'max_mb': self.max_memory_mb,
            'models_count': len(self._models),
            'shared_components_count': len(self._shared_components),
        }

    def clear(self):
        """Clear all models and components from pool."""
        for model_name in list(self._models.keys()):
            self.release_model(model_name)

        # Force cleanup of any remaining components
        for comp_type in list(self._shared_components.keys()):
            self.release_shared_component(comp_type)

        logger.info("Cleared model pool")