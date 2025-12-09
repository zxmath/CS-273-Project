"""
Model registry for managing different model architectures.

Models are registered using decorators and can be retrieved by name.
"""

from typing import Dict, Callable, Any, Optional, List
from .base import BaseTrajectoryModel


# Global registry of model factories
MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    """
    Decorator to register a model factory function.
    
    Usage:
        @register_model('sttransformer')
        def create_sttransformer(config, **kwargs):
            return STTransformer(...)
    
    Args:
        name: Model name/identifier
    """
    def decorator(factory_func: Callable[[Any], BaseTrajectoryModel]):
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = factory_func
        return factory_func
    return decorator


def get_model_factory(name: str) -> Optional[Callable]:
    """
    Get a model factory function by name.
    
    Args:
        name: Model name
        
    Returns:
        Factory function or None if not found
    """
    return MODEL_REGISTRY.get(name)


def list_models() -> List[str]:
    """List all registered model names."""
    return list(MODEL_REGISTRY.keys())


# Register STTransformer (import to trigger registration)
def _register_builtin_models():
    """Register built-in models."""
    try:
        from ..model import STTransformer
        
        @register_model('sttransformer')
        def create_sttransformer(config, **kwargs):
            """Factory for STTransformer model."""
            return STTransformer(
                input_dim=kwargs.get('input_dim'),
                hidden_dim=config.HIDDEN_DIM,
                horizon=config.MAX_FUTURE_HORIZON,
                window_size=config.WINDOW_SIZE,
                n_heads=config.STTRANSFORMER_N_HEADS,
                n_layers=config.STTRANSFORMER_N_LAYERS
            )
    except ImportError:
        pass


# Auto-register on import
_register_builtin_models()

