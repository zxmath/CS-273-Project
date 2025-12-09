"""
Backward compatibility module for model.py.

This file re-exports models and loss functions from their new organized locations
to maintain backward compatibility with existing imports.

New structure:
- models/sttransformer.py: STTransformer implementation
- loss.py: TemporalHuber loss function
"""

# Re-export for backward compatibility
from .models.sttransformer import STTransformer
from .loss import TemporalHuber

__all__ = ['STTransformer', 'TemporalHuber']
    