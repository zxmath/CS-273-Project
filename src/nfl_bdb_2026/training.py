"""
Pure training function for model training (no cross-validation).

This module provides a clean training function that can be used
for single train/val splits or within cross-validation loops.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import logging
from datetime import datetime
from typing import Optional, Callable, Dict, Any, Tuple

from .util import prepare_targets, compute_val_rmse


def train_model(
    model: nn.Module,
    X_train: list,
    y_train_dx: list,
    y_train_dy: list,
    X_val: list,
    y_val_dx: list,
    y_val_dy: list,
    config: Any,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    enable_logging: bool = True,
    log_dir: Optional[Path] = None,
    log_interval: int = 10,
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train a model with scheduled sampling support for decoder models.
    
    For decoder models with scheduled sampling enabled, gradually reduces
    teacher forcing probability from start_prob to end_prob over decay_epochs.
    """
    """
    Train a model on training data and validate on validation data.
    
    This is a pure training function - it doesn't do cross-validation.
    It can be used standalone or called from within a CV loop.
    
    Args:
        model: PyTorch model to train (must be on correct device)
        X_train: List of training input sequences
        y_train_dx: List of training target dx arrays (variable length)
        y_train_dy: List of training target dy arrays (variable length)
        X_val: List of validation input sequences
        y_val_dx: List of validation target dx arrays (variable length)
        y_val_dy: List of validation target dy arrays (variable length)
        config: Config object with training hyperparameters:
            - BATCH_SIZE: Batch size
            - EPOCHS: Maximum number of epochs
            - PATIENCE: Early stopping patience
            - MAX_FUTURE_HORIZON: Maximum prediction horizon
            - DEVICE: Device to train on
        criterion: Loss function (if None, must be handled by model or elsewhere)
        optimizer: Optimizer (if None, AdamW with config.LEARNING_RATE will be created)
        scheduler: Learning rate scheduler (if None, ReduceLROnPlateau will be created)
        enable_logging: Whether to log training progress
        log_dir: Directory to save logs and plots (if None, no files saved)
        log_interval: Log every N epochs
        
    Returns:
        model: Trained model (best state restored)
        history: Dictionary with training history:
            - train_loss: List of training losses
            - val_loss: List of validation losses
            - train_rmse: List of training RMSE (if computed)
            - val_rmse: List of validation RMSE (if computed)
            - learning_rates: List of learning rates per epoch
    """
    device = config.DEVICE
    horizon = config.MAX_FUTURE_HORIZON
    
    # Setup logging with timestamp
    logger = None
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if enable_logging:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"training_{timestamp_str}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
    
    # Setup loss function, optimizer, scheduler if not provided
    if criterion is None:
        from .loss import TemporalHuber
        criterion = TemporalHuber(delta=0.5, time_decay=0.03)
    
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-5
        )
    
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, verbose=False
        )
    
    # Prepare batches
    train_batches = []
    for i in range(0, len(X_train), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_train))
        bx = torch.tensor(np.stack(X_train[i:end]).astype(np.float32))
        by, bm = prepare_targets(
            [y_train_dx[j] for j in range(i, end)],
            [y_train_dy[j] for j in range(i, end)],
            horizon
        )
        train_batches.append((bx, by, bm))
    
    val_batches = []
    for i in range(0, len(X_val), config.BATCH_SIZE):
        end = min(i + config.BATCH_SIZE, len(X_val))
        bx = torch.tensor(np.stack(X_val[i:end]).astype(np.float32))
        by, bm = prepare_targets(
            [y_val_dx[j] for j in range(i, end)],
            [y_val_dy[j] for j in range(i, end)],
            horizon
        )
        val_batches.append((bx, by, bm))
    
    # Training loop
    best_loss, best_state, bad = float('inf'), None, 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': [],
        'learning_rates': []
    }
    
    # Setup scheduled sampling for decoder models
    use_scheduled_sampling = False
    ss_start_prob = 1.0
    ss_end_prob = 0.0
    ss_decay_epochs = config.EPOCHS // 2
    
    if hasattr(model, 'set_teacher_forcing_prob'):
        use_scheduled_sampling = getattr(config, 'STTRANSFORMER_DECODER_USE_SCHEDULED_SAMPLING', False)
        if use_scheduled_sampling:
            ss_start_prob = getattr(config, 'STTRANSFORMER_DECODER_SS_START_PROB', 1.0)
            ss_end_prob = getattr(config, 'STTRANSFORMER_DECODER_SS_END_PROB', 0.0)
            ss_decay_epochs = getattr(config, 'STTRANSFORMER_DECODER_SS_DECAY_EPOCHS', config.EPOCHS // 2)
            ss_decay_epochs = min(ss_decay_epochs, config.EPOCHS)
            if logger:
                logger.info(f"Scheduled sampling enabled: {ss_start_prob:.2f} → {ss_end_prob:.2f} over {ss_decay_epochs} epochs")
    
    if logger:
        logger.info(f"Starting training for {config.EPOCHS} epochs")
        logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
    
    for epoch in range(1, config.EPOCHS + 1):
        # Calculate teacher forcing probability for scheduled sampling
        if use_scheduled_sampling:
            if epoch <= ss_decay_epochs:
                # Linear decay from start_prob to end_prob
                progress = (epoch - 1) / max(1, ss_decay_epochs - 1)
                tf_prob = ss_start_prob - progress * (ss_start_prob - ss_end_prob)
            else:
                # After decay period, use end probability
                tf_prob = ss_end_prob
            model.set_teacher_forcing_prob(tf_prob)
        
        # Training phase
        model.train()
        train_losses = []
        try:
            for bx, by, bm in train_batches:
                bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                
                # Support teacher forcing for decoder models
                if hasattr(model, 'set_future_inputs'):
                    model.set_future_inputs(by)
                
                pred = model(bx)
                loss = criterion(pred, by, bm)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
        except Exception as e:
            error_msg = f"Error during training forward pass at epoch {epoch}: {type(e).__name__}: {e}"
            if logger:
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
            print(f"\n - {error_msg}")
            import traceback
            traceback.print_exc()
            raise
        
        # Validation phase
        model.eval()
        
        # Clear future_inputs for decoder models to ensure autoregressive mode
        if hasattr(model, 'clear_future_inputs'):
            model.clear_future_inputs()
        
        val_losses = []
        try:
            with torch.no_grad():
                for bx, by, bm in val_batches:
                    bx, by, bm = bx.to(device), by.to(device), bm.to(device)
                    pred = model(bx)
                    val_losses.append(criterion(pred, by, bm).item())
        except Exception as e:
            error_msg = f"Error during validation forward pass at epoch {epoch}: {type(e).__name__}: {e}"
            if logger:
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
            print(f"\n - {error_msg}")
            import traceback
            traceback.print_exc()
            raise
        
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(current_lr)
        
        # Compute RMSE periodically or at final epoch
        should_compute_rmse = (epoch % log_interval == 0) or (epoch == config.EPOCHS)
        
        if should_compute_rmse:
            train_rmse = compute_val_rmse(
                model, X_train, y_train_dx, y_train_dy,
                horizon, device, config.BATCH_SIZE
            )
            val_rmse = compute_val_rmse(
                model, X_val, y_val_dx, y_val_dy,
                horizon, device, config.BATCH_SIZE
            )
            history['train_rmse'].append((epoch, train_rmse))
            history['val_rmse'].append((epoch, val_rmse))
            
            if logger:
                logger.info(
                    f"Epoch {epoch}/{config.EPOCHS}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}, "
                    f"lr={current_lr:.6f}"
                )
        else:
            if logger:
                logger.info(
                    f"Epoch {epoch}/{config.EPOCHS}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"lr={current_lr:.6f}"
                )
        
        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= config.PATIENCE:
                if logger:
                    logger.info(f"Early stopping at epoch {epoch}")
                
                # Compute RMSE at early stopping if not already computed
                if not should_compute_rmse:
                    train_rmse = compute_val_rmse(
                        model, X_train, y_train_dx, y_train_dy,
                        horizon, device, config.BATCH_SIZE
                    )
                    val_rmse = compute_val_rmse(
                        model, X_val, y_val_dx, y_val_dy,
                        horizon, device, config.BATCH_SIZE
                    )
                    history['train_rmse'].append((epoch, train_rmse))
                    history['val_rmse'].append((epoch, val_rmse))
                
                break
    
    # Restore best model state
    if best_state:
        model.load_state_dict(best_state)
        if logger:
            logger.info(f"Training completed. Best val_loss: {best_loss:.4f}")
    
    return model, history

