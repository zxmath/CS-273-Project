"""
Utility functions for training and evaluation.

Key functions:
- set_seed: Set random seeds for reproducibility
- prepare_targets: Pad variable-length targets for batching
- compute_val_rmse: Compute RMSE for validation/evaluation
"""

import torch
import numpy as np
import random
import os


def set_seed(seed=42):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_targets(batch_dx, batch_dy, max_h):
    """
    Prepare variable-length targets for batching by padding to fixed size.
    
    WHY THIS IS NEEDED:
    Different players/plays have different prediction horizons (different number
    of future frames to predict). To batch them together, we need to:
    1. Pad all targets to the same length (max_horizon)
    2. Create masks to indicate which timesteps are valid vs padding
    
    This is architecture-agnostic - any model dealing with variable-length
    sequences needs this preprocessing step.
    
    Args:
        batch_dx: List of arrays, each containing dx values for one sample
        batch_dy: List of arrays, each containing dy values for one sample
        max_h: Maximum horizon length to pad to
        
    Returns:
        targets: (batch_size, horizon, 2) tensor of padded (dx, dy) targets
        masks: (batch_size, horizon) tensor of masks (1.0 = valid, 0.0 = padding)
    
    Example:
        batch_dx = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]  # lengths 2 and 3
        batch_dy = [np.array([0.5, 1.5]), np.array([1.5, 2.5, 3.5])]
        max_h = 5
        # Returns padded targets shape (2, 5, 2) and masks shape (2, 5)
    """
    tensors_x, tensors_y, masks = [], [], []
    
    for dx, dy in zip(batch_dx, batch_dy):
        L = len(dx)
        # Pad with zeros to max_horizon
        padded_x = np.pad(dx, (0, max_h - L), constant_values=0).astype(np.float32)
        padded_y = np.pad(dy, (0, max_h - L), constant_values=0).astype(np.float32)
        # Create mask: 1.0 for valid timesteps, 0.0 for padding
        mask = np.zeros(max_h, dtype=np.float32)
        mask[:L] = 1.0
        
        tensors_x.append(torch.tensor(padded_x))
        tensors_y.append(torch.tensor(padded_y))
        masks.append(torch.tensor(mask))
    
    # Stack: (batch_size, horizon, 2) for targets
    targets = torch.stack([torch.stack(tensors_x), torch.stack(tensors_y)], dim=-1)
    # Stack: (batch_size, horizon) for masks
    return targets, torch.stack(masks)


def compute_val_rmse(model, X_val, y_val_dx, y_val_dy, horizon, device, batch_size=256):
    """
    Compute validation RMSE (Root Mean Squared Error) for 2D trajectory prediction.
    
    Computes the average Euclidean distance error between predicted and actual
    trajectories across all validation samples.
    
    This function:
    - Works with any model that outputs (batch, horizon, 2) predictions
    - Handles variable-length sequences correctly
    - Returns RMSE in the same units as the data (yards)
    
    Args:
        model: Trained model (must support .eval() and output (batch, horizon, 2))
        X_val: List of input sequences (will be converted to tensor)
        y_val_dx: List of target dx arrays (variable length per sample)
        y_val_dy: List of target dy arrays (variable length per sample)
        horizon: Maximum prediction horizon
        device: Device to run inference on
        batch_size: Batch size for evaluation
        
    Returns:
        RMSE as a float (average Euclidean distance error in yards)
    """
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            end = min(i + batch_size, len(X_val))
            bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32)).to(device)
            
            # Get predictions: (batch, horizon, 2)
            pred = model(bx).cpu().numpy()
            
            # Process each sample in the batch
            for j, idx in enumerate(range(i, end)):
                dx_true = y_val_dx[idx]
                dy_true = y_val_dy[idx]
                n_steps = len(dx_true)
                
                # Extract predictions for this sample (only valid timesteps)
                dx_pred = pred[j, :n_steps, 0]
                dy_pred = pred[j, :n_steps, 1]
                
                # Compute Euclidean distance for each time step
                sq_errors = (dx_pred - dx_true) ** 2 + (dy_pred - dy_true) ** 2
                all_errors.extend(sq_errors)
    
    # Return RMSE
    return np.sqrt(np.mean(all_errors))
