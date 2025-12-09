"""
Loss functions for trajectory prediction.
"""

import torch
import torch.nn as nn


class TemporalHuber(nn.Module):
    """
    Temporal Huber Loss with time decay for trajectory prediction.
    
    This loss function:
    - Uses Huber loss (L2 for small errors, L1 for large errors) for robustness
    - Applies time decay to weight earlier predictions more heavily
    - Uses masks to handle variable-length sequences
    
    Args:
        delta: Threshold for switching between L2 and L1 loss
        time_decay: Exponential decay factor for time weighting
    """
    def __init__(self, delta=0.5, time_decay=0.03):
        super().__init__()
        self.delta = delta
        self.time_decay = time_decay

    def forward(self, pred, target, mask):
        """
        Compute temporal Huber loss.
        
        Args:
            pred: (batch_size, horizon, 2) - predicted cumulative displacements
            target: (batch_size, horizon, 2) - target cumulative displacements
            mask: (batch_size, horizon) - mask for valid timesteps (1.0 = valid, 0.0 = padding)
            
        Returns:
            Scalar loss value
        """
        err = pred - target
        abs_err = torch.abs(err)
        huber = torch.where(
            abs_err <= self.delta,
            0.5 * err * err,
            self.delta * (abs_err - 0.5 * self.delta)
        )

        if self.time_decay > 0:
            L = pred.size(1)
            t = torch.arange(L, device=pred.device).float()
            weight = torch.exp(-self.time_decay * t).view(1, L, 1)
            huber = huber * weight
            mask = mask.unsqueeze(-1)

        return (huber * mask).sum() / (mask.sum() + 1e-8)

