"""
Linear models for trajectory prediction.

Models: Ridge, Lasso, ElasticNet
These are simple baseline models that are fast to train and interpretable.
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from .base import BaseTrajectoryModel
from .registry import register_model


@register_model('ridge')
def create_ridge(config, **kwargs):
    """Factory function for Ridge regression model."""
    return LinearModel(
        model_type='ridge',
        alpha=kwargs.get('alpha', 1.0),
        horizon=config.MAX_FUTURE_HORIZON,
        config=config
    )


@register_model('lasso')
def create_lasso(config, **kwargs):
    """Factory function for Lasso regression model."""
    return LinearModel(
        model_type='lasso',
        alpha=kwargs.get('alpha', 0.1),
        horizon=config.MAX_FUTURE_HORIZON,
        config=config
    )


@register_model('elasticnet')
def create_elasticnet(config, **kwargs):
    """Factory function for ElasticNet regression model."""
    return LinearModel(
        model_type='elasticnet',
        alpha=kwargs.get('alpha', 0.1),
        l1_ratio=kwargs.get('l1_ratio', 0.5),
        horizon=config.MAX_FUTURE_HORIZON,
        config=config
    )


class LinearModel(BaseTrajectoryModel):
    """
    Linear model wrapper for trajectory prediction.
    
    Predicts all future timesteps simultaneously using multi-output regression.
    """
    
    def __init__(self, model_type='ridge', alpha=1.0, l1_ratio=0.5, horizon=94, config=None):
        super().__init__(config)
        self.model_type = model_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.horizon = horizon
        self.model_dx = None  # Model for x-direction
        self.model_dy = None  # Model for y-direction
        
        # Create base model
        if model_type == 'ridge':
            base_model = Ridge(alpha=alpha, max_iter=1000, random_state=42)
        elif model_type == 'lasso':
            base_model = Lasso(alpha=alpha, max_iter=1000, random_state=42)
        elif model_type == 'elasticnet':
            base_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Multi-output: predict all timesteps at once
        # Each output is one timestep, so we have horizon * 2 outputs (dx and dy)
        self.model_dx = MultiOutputRegressor(base_model, n_jobs=1)
        self.model_dy = MultiOutputRegressor(base_model, n_jobs=1)
    
    def fit(self, X, y_dx, y_dy=None):
        """
        Fit the model.
        
        Args:
            X: Flattened input features (n_samples, window_size * n_features)
            y_dx: Target dx values (n_samples, horizon) - padded to max horizon
            y_dy: Target dy values (n_samples, horizon) - padded to max horizon (optional for compatibility)
        """
        # Flatten to 2D if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Handle single target for compatibility
        if y_dy is None:
            # Assume y_dx is actually a tuple/list of (y_dx, y_dy)
            if isinstance(y_dx, (tuple, list)) and len(y_dx) == 2:
                y_dx, y_dy = y_dx
            else:
                raise ValueError("y_dy must be provided")
        
        # Fit separate models for dx and dy
        self.model_dx.fit(X, y_dx)
        self.model_dy.fit(X, y_dy)
    
    def predict(self, X):
        """
        Predict trajectories.
        
        Args:
            X: Input features (n_samples, window_size, n_features) or (n_samples, window_size * n_features)
            
        Returns:
            Predictions as (n_samples, horizon, 2) array
        """
        # Flatten to 2D if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Predict dx and dy
        pred_dx = self.model_dx.predict(X)  # (n_samples, horizon) - already cumulative
        pred_dy = self.model_dy.predict(X)  # (n_samples, horizon) - already cumulative
        
        # Stack to (n_samples, horizon, 2)
        # Note: MultiOutputRegressor learns cumulative displacements directly
        # (one regressor per timestep), so no cumsum needed
        pred = np.stack([pred_dx, pred_dy], axis=-1)
        
        return pred
    
    def forward(self, x):
        """PyTorch-style forward pass (wrapper for predict)."""
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
        return self.predict(x)

