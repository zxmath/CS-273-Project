"""
Training utilities for sklearn-style models (linear, tree, etc.).

These models use fit/predict interface rather than PyTorch training loops.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor

from .util import prepare_targets, compute_val_rmse


def prepare_data_for_sklearn(
    sequences: List[np.ndarray],
    targets_dx: List[np.ndarray],
    targets_dy: List[np.ndarray],
    horizon: int,
    flatten: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare sequences for sklearn models.
    
    Args:
        sequences: List of input sequences (n_samples, window_size, features)
        targets_dx: List of target dx arrays (variable length)
        targets_dy: List of target dy arrays (variable length)
        horizon: Maximum prediction horizon
        flatten: If True, flatten sequences to (n_samples, window_size * features)
        
    Returns:
        X: Input features (n_samples, flattened_features)
        y_dx: Target dx (n_samples, horizon) - padded
        y_dy: Target dy (n_samples, horizon) - padded
        masks: Valid timesteps mask (n_samples, horizon)
    """
    # Stack sequences
    X = np.stack(sequences)  # (n_samples, window_size, features)
    
    # Flatten if needed
    if flatten:
        X = X.reshape(X.shape[0], -1)  # (n_samples, window_size * features)
    
    # Prepare targets with padding
    y_dx_padded = []
    y_dy_padded = []
    masks = []
    
    for dx, dy in zip(targets_dx, targets_dy):
        n_steps = len(dx)
        
        # Pad to horizon
        dx_pad = np.pad(dx, (0, horizon - n_steps), constant_values=0).astype(np.float32)
        dy_pad = np.pad(dy, (0, horizon - n_steps), constant_values=0).astype(np.float32)
        
        # Create mask
        mask = np.zeros(horizon, dtype=np.float32)
        mask[:n_steps] = 1.0
        
        y_dx_padded.append(dx_pad)
        y_dy_padded.append(dy_pad)
        masks.append(mask)
    
    y_dx = np.stack(y_dx_padded)  # (n_samples, horizon)
    y_dy = np.stack(y_dy_padded)  # (n_samples, horizon)
    masks = np.stack(masks)  # (n_samples, horizon)
    
    return X, y_dx, y_dy, masks


def train_sklearn_model(
    model,
    X_train: np.ndarray,
    y_train_dx: np.ndarray,
    y_train_dy: np.ndarray,
    X_val: np.ndarray,
    y_val_dx: np.ndarray,
    y_val_dy: np.ndarray,
    masks_val: np.ndarray,
    masks_train: np.ndarray = None,
    scaler: Any = None,
    use_scaling: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a sklearn-style model.
    
    Args:
        model: Model with fit() and predict() methods
        X_train: Training features
        y_train_dx: Training dx targets
        y_train_dy: Training dy targets
        X_val: Validation features
        y_val_dx: Validation dx targets
        y_val_dy: Validation dy targets
        masks_val: Validation masks
        scaler: Optional scaler (if None, creates new StandardScaler)
        use_scaling: Whether to scale features
        
    Returns:
        Trained model and history dictionary
    """
    # Scale features if needed
    if scaler is None and use_scaling:
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
    elif use_scaling:
        X_train_sc = scaler.transform(X_train)
        X_val_sc = scaler.transform(X_val)
    else:
        X_train_sc = X_train
        X_val_sc = X_val
        scaler = None
    
    # Fit model - models expect (X, y_dx, y_dy) signature
    if hasattr(model, 'fit'):
        # Check if model expects separate y_dx and y_dy
        import inspect
        sig = inspect.signature(model.fit)
        params = list(sig.parameters.keys())
        
        if 'y_dy' in params or len(params) >= 3:
            model.fit(X_train_sc, y_train_dx, y_train_dy)
        else:
            # Fallback: assume it can handle tuple
            model.fit(X_train_sc, (y_train_dx, y_train_dy))
    else:
        raise ValueError("Model must have fit() method")
    
    # Compute training loss/RMSE
    train_pred = model.predict(X_train_sc)  # (n_train, horizon, 2)
    train_errors = []
    
    # Use provided masks or create them from data
    if masks_train is None:
        # Create masks for training data (assume all timesteps up to horizon are valid)
        masks_train = np.ones((len(X_train), y_train_dx.shape[1]), dtype=np.float32)
    
    for i in range(len(X_train)):
        n_steps = int(masks_train[i].sum())
        if n_steps > 0:
            dx_true = y_train_dx[i, :n_steps]
            dy_true = y_train_dy[i, :n_steps]
            dx_pred = train_pred[i, :n_steps, 0]
            dy_pred = train_pred[i, :n_steps, 1]
            
            sq_errors = (dx_pred - dx_true) ** 2 + (dy_pred - dy_true) ** 2
            train_errors.extend(sq_errors)
    
    train_rmse = np.sqrt(np.mean(train_errors)) if train_errors else np.inf
    train_loss = train_rmse  # Use RMSE as loss for sklearn models
    
    # Compute validation loss
    val_pred = model.predict(X_val_sc)  # (n_val, horizon, 2)
    
    # Compute masked RMSE
    val_errors = []
    for i in range(len(X_val)):
        n_steps = int(masks_val[i].sum())
        if n_steps > 0:
            dx_true = y_val_dx[i, :n_steps]
            dy_true = y_val_dy[i, :n_steps]
            dx_pred = val_pred[i, :n_steps, 0]
            dy_pred = val_pred[i, :n_steps, 1]
            
            sq_errors = (dx_pred - dx_true) ** 2 + (dy_pred - dy_true) ** 2
            val_errors.extend(sq_errors)
    
    val_rmse = np.sqrt(np.mean(val_errors)) if val_errors else np.inf
    val_loss = val_rmse  # Use RMSE as loss for sklearn models
    
    history = {
        'train_loss': [train_loss],  # Actual training loss
        'val_loss': [val_loss],
        'train_rmse': [train_rmse],
        'val_rmse': [(1, val_rmse)],  # (epoch, rmse) format
        'learning_rates': [0.0],  # Not applicable
    }
    
    return model, scaler, history


def compute_val_rmse_sklearn(
    model: Any,
    X_val: List[np.ndarray],
    y_val_dx: List[np.ndarray],
    y_val_dy: List[np.ndarray],
    horizon: int,
    scaler: Any = None,
    use_scaling: bool = True
) -> float:
    """
    Compute validation RMSE for sklearn models.
    
    Args:
        model: Trained model
        X_val: List of validation sequences
        y_val_dx: List of validation dx targets
        y_val_dy: List of validation dy targets
        horizon: Maximum horizon
        scaler: Optional scaler
        use_scaling: Whether to scale features
        
    Returns:
        RMSE value
    """
    # Prepare data
    X_val_stacked = np.stack(X_val)
    
    if use_scaling and scaler:
        X_val_sc = scaler.transform(X_val_stacked.reshape(len(X_val), -1))
    else:
        X_val_sc = X_val_stacked.reshape(len(X_val), -1)
    
    # Predict
    predictions = model.predict(X_val_sc)  # (n_samples, horizon, 2)
    
    # Compute RMSE
    all_errors = []
    for i in range(len(X_val)):
        dx_true = y_val_dx[i]
        dy_true = y_val_dy[i]
        n_steps = len(dx_true)
        
        dx_pred = predictions[i, :n_steps, 0]
        dy_pred = predictions[i, :n_steps, 1]
        
        sq_errors = (dx_pred - dx_true) ** 2 + (dy_pred - dy_true) ** 2
        all_errors.extend(sq_errors)
    
    return np.sqrt(np.mean(all_errors)) if all_errors else np.inf

