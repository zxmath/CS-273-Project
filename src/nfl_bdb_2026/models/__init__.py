"""
Model registry for NFL trajectory prediction.

This module provides a unified interface for training different model architectures:
- STTransformer (Spatio-Temporal Transformer)
- GRU (Gated Recurrent Unit)
- Tree models (XGBoost, LightGBM, CatBoost)
- Linear models (Ridge, Lasso, ElasticNet)
- ARIMA (Time Series)
"""

from .base import BaseTrajectoryModel, ModelFactory
from .registry import MODEL_REGISTRY, register_model, get_model_factory

# Import models to trigger registration
try:
    from . import linear  # Registers ridge, lasso, elasticnet
except ImportError:
    pass

try:
    from . import tree  # Registers xgboost, lightgbm, catboost
except ImportError:
    pass

try:
    from . import arima  # Registers arima
except ImportError:
    pass

try:
    from . import sttransformer_model  # Registers sttransformer (encoder-only)
except ImportError:
    pass

try:
    from . import sttransformer_decoder_model  # Registers sttransformer_decoder
except ImportError:
    pass

__all__ = [
    'BaseTrajectoryModel',
    'ModelFactory',
    'MODEL_REGISTRY',
    'register_model',
    'get_model_factory',
]

