"""
ARIMA time series models for trajectory prediction.

Note: ARIMA is primarily designed for univariate time series and may not
perform well for this multi-variable spatial prediction task. Consider this
as an experimental baseline rather than a primary approach.
"""

import numpy as np
from typing import List, Tuple
from .base import BaseTrajectoryModel
from .registry import register_model

# Try importing ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False


@register_model('arima')
def create_arima(config, **kwargs):
    """Factory function for ARIMA model."""
    if not ARIMA_AVAILABLE:
        raise ImportError("statsmodels is not installed. Install with: pip install statsmodels")
    return ARIMAModel(
        order=kwargs.get('order', (2, 1, 2)),
        horizon=config.MAX_FUTURE_HORIZON,
        config=config
    )


class ARIMAModel(BaseTrajectoryModel):
    """
    ARIMA model for trajectory prediction.
    
    Fits separate ARIMA models for x and y coordinates.
    Note: This approach ignores spatial context and may not perform well.
    """
    
    def __init__(self, order=(2, 1, 2), horizon=94, config=None):
        super().__init__(config)
        self.order = order
        self.horizon = horizon
        self.models_x = []  # List of ARIMA models (one per sample)
        self.models_y = []
        self.is_fitted = False
        
        # Store training data for refitting per sample
        self._train_sequences = None
        self._train_targets_dx = None
        self._train_targets_dy = None
    
    def fit(self, X, y_dx, y_dy=None):
        """
        Fit ARIMA models.
        
        Note: ARIMA is simplified - uses linear extrapolation baseline.
        Full ARIMA per-sample fitting is too slow for this task.
        
        Args:
            X: Input sequences (n_samples, window_size, features) or flattened
            y_dx: Target dx (n_samples, horizon) - not used for baseline
            y_dy: Target dy (n_samples, horizon) - not used for baseline
        """
        # ARIMA uses a simple baseline prediction (linear extrapolation)
        # No actual ARIMA fitting needed for the simplified version
        self.is_fitted = True
    
    def _fit_single_arima(self, ts_data: np.ndarray) -> Tuple:
        """
        Fit ARIMA model to a single time series.
        
        Args:
            ts_data: Time series data (1D array)
            
        Returns:
            (model_x, model_y) tuple of fitted ARIMA models
        """
        if not ARIMA_AVAILABLE:
            raise ImportError("statsmodels not available")
        
        try:
            model = ARIMA(ts_data, order=self.order)
            fitted_model = model.fit()
            return fitted_model
        except Exception as e:
            # Fallback to simple autoregressive model
            # Use last value as prediction
            return None
    
    def predict(self, X) -> np.ndarray:
        """
        Predict using ARIMA.
        
        This is a simplified implementation that uses the last values
        from the sequence as a baseline, since fitting ARIMA per-sample
        is computationally expensive.
        
        Args:
            X: Input sequences - can be list of arrays or numpy array (n_samples, window_size, features)
            
        Returns:
            Predictions as (n_samples, horizon, 2) array
        """
        # Convert to list if needed
        if isinstance(X, np.ndarray):
            sequences = [X[i] for i in range(len(X))]
        elif isinstance(X, list):
            sequences = X
        else:
            raise TypeError(f"Unexpected input type: {type(X)}")
        
        n_samples = len(sequences)
        predictions = np.zeros((n_samples, self.horizon, 2))
        
        for i, seq in enumerate(sequences):
            # Extract x and y from sequence (assume first two features are x, y)
            if seq.shape[1] >= 2:
                ts_x = seq[:, 0]  # x coordinate over time
                ts_y = seq[:, 1]  # y coordinate over time
                
                # Simple baseline: predict based on recent trend
                # Use linear extrapolation from last few points
                window = min(5, len(ts_x))
                if window >= 2:
                    # Linear trend
                    x_trend = (ts_x[-1] - ts_x[-window]) / (window - 1)
                    y_trend = (ts_y[-1] - ts_y[-window]) / (window - 1)
                    
                    # Predict future positions (cumulative from last position)
                    last_x = ts_x[-1]
                    last_y = ts_y[-1]
                    
                    for t in range(self.horizon):
                        predictions[i, t, 0] = last_x + x_trend * (t + 1)
                        predictions[i, t, 1] = last_y + y_trend * (t + 1)
                else:
                    # Fallback: constant prediction
                    predictions[i, :, 0] = ts_x[-1]
                    predictions[i, :, 1] = ts_y[-1]
        
        return predictions
    
    def forward(self, x):
        """PyTorch-style forward pass."""
        # Convert tensor to list of arrays
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
        
        sequences = [x[i] for i in range(len(x))]
        return self.predict(sequences)

