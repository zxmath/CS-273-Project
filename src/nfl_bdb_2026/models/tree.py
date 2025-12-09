"""
Tree-based models for trajectory prediction.

Models: XGBoost, LightGBM, CatBoost
These are powerful gradient boosting models that are fast and don't require feature scaling.
"""

import numpy as np
from .base import BaseTrajectoryModel
from .registry import register_model

# Try importing tree libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


@register_model('xgboost')
def create_xgboost(config, **kwargs):
    """Factory function for XGBoost model."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Install with: pip install xgboost")
    # Use Config values as defaults, override with kwargs
    model_kwargs = {
        'n_estimators': kwargs.get('n_estimators', config.TREE_N_ESTIMATORS),
        'max_depth': kwargs.get('max_depth', config.TREE_MAX_DEPTH),
        'learning_rate': kwargs.get('learning_rate', config.TREE_LEARNING_RATE),
        'subsample': kwargs.get('subsample', getattr(config, 'TREE_SUBSAMPLE', 1.0)),
        'colsample_bytree': kwargs.get('colsample_bytree', getattr(config, 'TREE_COLSAMPLE_BYTREE', 1.0)),
        'min_child_weight': kwargs.get('min_child_weight', getattr(config, 'TREE_MIN_CHILD_WEIGHT', 1)),
        'reg_alpha': kwargs.get('reg_alpha', getattr(config, 'TREE_REG_ALPHA', 0)),
        'reg_lambda': kwargs.get('reg_lambda', getattr(config, 'TREE_REG_LAMBDA', 1)),
        'early_stopping_rounds': kwargs.get('early_stopping_rounds', getattr(config, 'TREE_EARLY_STOPPING_ROUNDS', 0)),
    }
    model_kwargs.update(kwargs)  # Allow additional kwargs to override
    return TreeModel(
        model_type='xgboost',
        horizon=config.MAX_FUTURE_HORIZON,
        config=config,
        **model_kwargs
    )


@register_model('lightgbm')
def create_lightgbm(config, **kwargs):
    """Factory function for LightGBM model."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Install with: pip install lightgbm")
    # Use Config values as defaults, override with kwargs
    model_kwargs = {
        'n_estimators': kwargs.get('n_estimators', config.TREE_N_ESTIMATORS),
        'max_depth': kwargs.get('max_depth', config.TREE_MAX_DEPTH),
        'learning_rate': kwargs.get('learning_rate', config.TREE_LEARNING_RATE),
        'subsample': kwargs.get('subsample', getattr(config, 'TREE_SUBSAMPLE', 1.0)),
        'colsample_bytree': kwargs.get('colsample_bytree', getattr(config, 'TREE_COLSAMPLE_BYTREE', 1.0)),
        'min_child_samples': kwargs.get('min_child_samples', getattr(config, 'TREE_MIN_CHILD_WEIGHT', 20)),  # LightGBM uses min_child_samples
        'lambda_l1': kwargs.get('lambda_l1', getattr(config, 'TREE_REG_ALPHA', 0)),  # LightGBM uses lambda_l1
        'lambda_l2': kwargs.get('lambda_l2', getattr(config, 'TREE_REG_LAMBDA', 1.0)),  # LightGBM uses lambda_l2
        'early_stopping_rounds': kwargs.get('early_stopping_rounds', getattr(config, 'TREE_EARLY_STOPPING_ROUNDS', 0)),
    }
    model_kwargs.update(kwargs)  # Allow additional kwargs to override
    return TreeModel(
        model_type='lightgbm',
        horizon=config.MAX_FUTURE_HORIZON,
        config=config,
        **model_kwargs
    )


@register_model('catboost')
def create_catboost(config, **kwargs):
    """Factory function for CatBoost model."""
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost is not installed. Install with: pip install catboost")
    # Use Config values as defaults, override with kwargs
    model_kwargs = {
        'n_estimators': kwargs.get('n_estimators', config.TREE_N_ESTIMATORS),
        'max_depth': kwargs.get('max_depth', config.TREE_MAX_DEPTH),
        'learning_rate': kwargs.get('learning_rate', config.TREE_LEARNING_RATE),
        'subsample': kwargs.get('subsample', getattr(config, 'TREE_SUBSAMPLE', 1.0)),
        'colsample_bytree': kwargs.get('colsample_bytree', getattr(config, 'TREE_COLSAMPLE_BYTREE', 1.0)),
        # Note: CatBoost uses different parameter names for regularization
    }
    model_kwargs.update(kwargs)  # Allow additional kwargs to override
    return TreeModel(
        model_type='catboost',
        horizon=config.MAX_FUTURE_HORIZON,
        config=config,
        **model_kwargs
    )


class TreeModel(BaseTrajectoryModel):
    """
    Tree-based model wrapper for trajectory prediction.
    
    Uses gradient boosting trees to predict all future timesteps simultaneously.
    Supports GPU acceleration for XGBoost and LightGBM.
    """
    
    def __init__(self, model_type='xgboost', horizon=94, config=None, **kwargs):
        super().__init__(config)
        self.model_type = model_type
        self.horizon = horizon
        self.model_dx = None
        self.model_dy = None
        self.kwargs = kwargs
        self.early_stopping_rounds = kwargs.get('early_stopping_rounds', 
                                                 getattr(config, 'TREE_EARLY_STOPPING_ROUNDS', 0) if config else 0)
        
        # Determine device for GPU acceleration
        # Use GPU_ID directly from config
        use_gpu = False
        tree_device = 'cpu'
        gpu_id = 0
        
        if config and hasattr(config, 'GPU_ID'):
            gpu_id = config.GPU_ID
            # Check if CUDA is available
            try:
                import torch
                if torch.cuda.is_available():
                    use_gpu = True
                    tree_device = f'cuda:{gpu_id}'
            except ImportError:
                pass  # torch not available, skip GPU
        
        # Create base model with Config values and GPU support
        if model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not available")
            
            base_params = {
                'objective': 'reg:squarederror',
                'n_estimators': kwargs.get('n_estimators', config.TREE_N_ESTIMATORS if config else 100),
                'max_depth': kwargs.get('max_depth', config.TREE_MAX_DEPTH if config else 6),
                'learning_rate': kwargs.get('learning_rate', config.TREE_LEARNING_RATE if config else 0.1),
                'subsample': kwargs.get('subsample', getattr(config, 'TREE_SUBSAMPLE', 1.0) if config else 1.0),
                'colsample_bytree': kwargs.get('colsample_bytree', getattr(config, 'TREE_COLSAMPLE_BYTREE', 1.0) if config else 1.0),
                'min_child_weight': kwargs.get('min_child_weight', getattr(config, 'TREE_MIN_CHILD_WEIGHT', 1) if config else 1),
                'reg_alpha': kwargs.get('reg_alpha', getattr(config, 'TREE_REG_ALPHA', 0) if config else 0),
                'reg_lambda': kwargs.get('reg_lambda', getattr(config, 'TREE_REG_LAMBDA', 1) if config else 1),
                'random_state': config.SEED if config else 42,
                'verbosity': 0,
            }
            
            # Enable GPU for XGBoost if available
            if use_gpu:
                # XGBoost 2.0+ supports direct GPU via device parameter
                base_params['tree_method'] = 'hist'  # GPU-compatible method
                base_params['device'] = tree_device  # e.g., 'cuda:5'
                print(f"   - XGBoost will use GPU: {tree_device}")
            else:
                base_params['n_jobs'] = -1  # Use all CPU cores if no GPU
                print(f"   - XGBoost will use CPU (all cores)")
            
            base_model = xgb.XGBRegressor(**base_params)
            
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not available")
            
            base_params = {
                'objective': 'regression',
                'n_estimators': kwargs.get('n_estimators', config.TREE_N_ESTIMATORS if config else 100),
                'max_depth': kwargs.get('max_depth', config.TREE_MAX_DEPTH if config else 6),
                'learning_rate': kwargs.get('learning_rate', config.TREE_LEARNING_RATE if config else 0.1),
                'subsample': kwargs.get('subsample', getattr(config, 'TREE_SUBSAMPLE', 1.0) if config else 1.0),
                'colsample_bytree': kwargs.get('colsample_bytree', getattr(config, 'TREE_COLSAMPLE_BYTREE', 1.0) if config else 1.0),
                'min_child_samples': kwargs.get('min_child_samples', getattr(config, 'TREE_MIN_CHILD_WEIGHT', 20) if config else 20),  # LightGBM equivalent
                'lambda_l1': kwargs.get('lambda_l1', getattr(config, 'TREE_REG_ALPHA', 0) if config else 0),
                'lambda_l2': kwargs.get('lambda_l2', getattr(config, 'TREE_REG_LAMBDA', 1.0) if config else 1.0),
                'random_state': config.SEED if config else 42,
                'verbosity': -1,
            }
            
            # Enable GPU for LightGBM if available
            if use_gpu:
                base_params['device'] = 'gpu'
                base_params['gpu_platform_id'] = 0
                base_params['gpu_device_id'] = gpu_id
                print(f"   - LightGBM will use GPU: {tree_device} (device_id={gpu_id})")
            else:
                base_params['n_jobs'] = -1  # Use all CPU cores
                print(f"   - LightGBM will use CPU (all cores)")
            
            base_model = lgb.LGBMRegressor(**base_params)
            
        elif model_type == 'catboost':
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not available")
            
            base_params = {
                'iterations': kwargs.get('n_estimators', config.TREE_N_ESTIMATORS if config else 100),
                'depth': kwargs.get('max_depth', config.TREE_MAX_DEPTH if config else 6),
                'learning_rate': kwargs.get('learning_rate', config.TREE_LEARNING_RATE if config else 0.1),
                'random_seed': config.SEED if config else 42,
                'verbose': False,
            }
            
            # CatBoost GPU support is limited and can be problematic
            # Always use CPU for CatBoost to ensure stability
            print(f"   - CatBoost will use CPU (GPU support disabled for stability)")
            
            base_model = cb.CatBoostRegressor(**base_params)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Use MultiOutputRegressor for multi-target prediction
        # Note: MultiOutputRegressor doesn't support GPU directly, but each underlying model does
        from sklearn.multioutput import MultiOutputRegressor
        self.model_dx = MultiOutputRegressor(base_model, n_jobs=1)  # n_jobs=1 to avoid conflicts with GPU
        self.model_dy = MultiOutputRegressor(base_model, n_jobs=1)
    
    def fit(self, X, y_dx, y_dy=None):
        """
        Fit the model.
        
        Args:
            X: Flattened input features (n_samples, window_size * n_features)
            y_dx: Target dx values (n_samples, horizon)
            y_dy: Target dy values (n_samples, horizon) (optional for compatibility)
        """
        # Flatten to 2D if needed
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        
        # Handle single target for compatibility
        if y_dy is None:
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

