"""
Base classes for trajectory prediction models.

All models must inherit from BaseTrajectoryModel and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
import torch.nn as nn


class BaseTrajectoryModel(ABC):
    """
    Abstract base class for trajectory prediction models.
    
    All models must implement:
    - forward(): Forward pass for neural network models
    - predict(): Prediction method (may wrap forward for neural nets)
    """
    
    def __init__(self, config: Any):
        """
        Initialize model with configuration.
        
        Args:
            config: Configuration object with model hyperparameters
        """
        self.config = config
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass (for PyTorch models).
        
        Args:
            x: Input tensor
            
        Returns:
            Predictions tensor
        """
        pass
    
    def predict(self, x):
        """
        Make predictions (wrapper around forward for neural networks,
        or custom implementation for tree/linear models).
        
        Args:
            x: Input data
            
        Returns:
            Predictions
        """
        if hasattr(self, 'forward'):
            return self.forward(x)
        raise NotImplementedError("Model must implement either forward() or predict()")
    
    def eval(self):
        """Set model to evaluation mode."""
        if isinstance(self, nn.Module):
            super().eval()
    
    def train(self, mode=True):
        """Set model to training mode."""
        if isinstance(self, nn.Module):
            super().train(mode)


class ModelFactory:
    """
    Factory class for creating model instances.
    
    Usage:
        factory = ModelFactory(model_type='sttransformer', config=config)
        model = factory.create(input_dim=159, ...)
    """
    
    def __init__(self, model_type: str, config: Any):
        """
        Initialize factory.
        
        Args:
            model_type: Type of model to create (e.g., 'sttransformer', 'gru', 'xgboost')
            config: Configuration object
        """
        self.model_type = model_type
        self.config = config
    
    def create(self, **kwargs) -> BaseTrajectoryModel:
        """
        Create a model instance.
        
        Args:
            **kwargs: Model-specific initialization arguments
            
        Returns:
            Model instance
        """
        from .registry import get_model_factory
        
        factory_func = get_model_factory(self.model_type)
        if factory_func is None:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return factory_func(self.config, **kwargs)

