"""
Universal cross-validation framework for model training.

This module provides a flexible cross-validation function that works
with any model architecture and can be easily extended.
"""

import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List, Tuple
import logging
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm

from .training import train_model
from .util import compute_val_rmse


def cross_validate(
    model_factory: Callable[[], Any],
    sequences: list,
    targets_dx: list,
    targets_dy: list,
    groups: np.ndarray,
    config: Any,
    n_folds: Optional[int] = None,
    scaler_factory: Optional[Callable] = None,
    enable_logging: bool = True,
    enable_plots: bool = True,
    output_dir: Optional[Path] = None,
    log_interval: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Perform cross-validation with GroupKFold splitting.
    
    This is a universal CV function that works with any model architecture.
    It handles:
    - GroupKFold splitting (by game_id to prevent data leakage)
    - Model training for each fold
    - Aggregation of results across folds
    - Logging and visualization
    
    Args:
        model_factory: Function that creates a new model instance
                      Must accept no arguments and return a PyTorch model
                      Example: lambda: STTransformer(input_dim=167, ...)
        sequences: List of input sequences (variable length lists)
        targets_dx: List of target dx arrays (same length as sequences)
        targets_dy: List of target dy arrays (same length as sequences)
        groups: Array of group IDs (e.g., game_id) for GroupKFold
        config: Config object with hyperparameters
        n_folds: Number of CV folds (defaults to config.N_FOLDS)
        scaler_factory: Function to create/return scaler for feature scaling
                       If None, StandardScaler from sklearn is used
        enable_logging: Whether to log CV progress
        enable_plots: Whether to generate and save plots
        output_dir: Directory to save logs and plots (if None, uses config.MODEL_DIR)
        
    Returns:
        Dictionary with CV results:
            - fold_losses: List of validation losses per fold
            - fold_rmses: List of validation RMSEs per fold
            - mean_loss: Average validation loss across folds
            - std_loss: Std dev of validation losses
            - mean_rmse: Average validation RMSE across folds
            - std_rmse: Std dev of validation RMSEs
            - fold_histories: List of training histories per fold
            - models: List of trained models (one per fold)
            - scalers: List of scalers (one per fold)
    """
    if n_folds is None:
        n_folds = config.N_FOLDS
    
    if output_dir is None:
        output_dir = config.MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging with timestamp
    logger = None
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if enable_logging:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        if output_dir:
            log_file = output_dir / f"cross_validation_{timestamp_str}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            logger.addHandler(file_handler)
    
    # Setup scaler factory
    if scaler_factory is None:
        from sklearn.preprocessing import StandardScaler
        scaler_factory = StandardScaler
    
    # GroupKFold splitting
    gkf = GroupKFold(n_splits=n_folds)
    
    # Storage for results
    fold_losses = []
    fold_rmses = []
    fold_histories = []
    models = []
    scalers = []
    
    if logger:
        logger.info(f"Starting {n_folds}-fold cross-validation")
        logger.info(f"Total samples: {len(sequences)}")
    
    # CV loop
    for fold, (tr_idx, va_idx) in enumerate(
        tqdm(gkf.split(sequences, groups=groups), total=n_folds, desc="CV Folds"),
        start=1
    ):
        if logger:
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold}/{n_folds}")
            logger.info(f"{'='*60}")
        
        # Split data
        X_train = [sequences[i] for i in tr_idx]
        X_val = [sequences[i] for i in va_idx]
        y_train_dx = [targets_dx[i] for i in tr_idx]
        y_val_dx = [targets_dx[i] for i in va_idx]
        y_train_dy = [targets_dy[i] for i in tr_idx]
        y_val_dy = [targets_dy[i] for i in va_idx]
        
        # Feature scaling
        scaler = scaler_factory()
        scaler.fit(np.vstack([s for s in X_train]))
        X_train_sc = [scaler.transform(s) for s in X_train]
        X_val_sc = [scaler.transform(s) for s in X_val]
        
        # Create fresh model for this fold
        model = model_factory().to(config.DEVICE)
        
        # Train model with error handling for individual folds
        fold_log_dir = output_dir / f"fold_{fold}" if enable_logging else None
        # Determine log_interval - use provided value or default to 10
        fold_log_interval = log_interval if log_interval is not None else 10
        # If epochs are very few, compute RMSE more frequently
        if config.EPOCHS <= 5:
            fold_log_interval = min(fold_log_interval, 2)
        
        try:
            trained_model, history = train_model(
                model=model,
                X_train=X_train_sc,
                y_train_dx=y_train_dx,
                y_train_dy=y_train_dy,
                X_val=X_val_sc,
                y_val_dx=y_val_dx,
                y_val_dy=y_val_dy,
                config=config,
                enable_logging=enable_logging,
                log_dir=fold_log_dir,
                log_interval=fold_log_interval,
            )
            
            # Compute final validation RMSE
            val_rmse = compute_val_rmse(
                trained_model, X_val_sc, y_val_dx, y_val_dy,
                config.MAX_FUTURE_HORIZON, config.DEVICE, config.BATCH_SIZE
            )
            
            # Get final validation loss from history
            val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
            
            fold_losses.append(val_loss)
            fold_rmses.append(val_rmse)
            fold_histories.append(history)
            models.append(trained_model)
            scalers.append(scaler)
            
            if logger:
                logger.info(f"Fold {fold} - Val Loss: {val_loss:.5f}, Val RMSE: {val_rmse:.5f}")
            
            # Generate fold-specific plots if enabled
            if enable_plots:
                _plot_fold_history(history, fold, output_dir, timestamp_str)
        
        except Exception as e:
            error_msg = f"Fold {fold} training failed: {type(e).__name__}: {e}"
            if logger:
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())
            print(f"   - {error_msg}")
            
            # Add placeholder values so we can continue with other folds
            fold_losses.append(float('inf'))
            fold_rmses.append(float('inf'))
            fold_histories.append({})
            models.append(None)  # None indicates failed fold
            scalers.append(None)
            
            # Continue to next fold instead of crashing
            continue
    
    # Aggregate results (filter out failed folds with inf values)
    valid_losses = [l for l in fold_losses if l != float('inf')]
    valid_rmses = [r for r in fold_rmses if r != float('inf')]
    
    if len(valid_losses) == 0:
        raise RuntimeError("All folds failed during training. No models were successfully trained.")
    
    mean_loss = np.mean(valid_losses)
    std_loss = np.std(valid_losses) if len(valid_losses) > 1 else 0.0
    mean_rmse = np.mean(valid_rmses)
    std_rmse = np.std(valid_rmses) if len(valid_rmses) > 1 else 0.0
    
    results = {
        'fold_losses': fold_losses,
        'fold_rmses': fold_rmses,
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'fold_histories': fold_histories,
        'models': models,
        'scalers': scalers,
    }
    
    if logger:
        logger.info(f"\n{'='*60}")
        logger.info("Cross-Validation Results")
        logger.info(f"{'='*60}")
        logger.info(f"Mean Val Loss: {mean_loss:.5f} ± {std_loss:.5f}")
        logger.info(f"Mean Val RMSE: {mean_rmse:.5f} ± {std_rmse:.5f}")
        logger.info(f"\nFold Results:")
        for i, (loss, rmse) in enumerate(zip(fold_losses, fold_rmses), 1):
            logger.info(f"  Fold {i}: Loss={loss:.5f}, RMSE={rmse:.5f}")
    
    # Generate aggregate plots if enabled
    if enable_plots:
        _plot_cv_summary(results, output_dir, timestamp_str)
    
    return results


def _plot_fold_history(history: Dict, fold: int, output_dir: Path, timestamp: str = ""):
    """Plot training history for a single fold."""
    # Determine number of subplots needed
    has_rmse = bool(history.get('val_rmse'))
    
    if has_rmse:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    else:
        # If no RMSE data, create 2x1 layout instead
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        # Convert to 2D array for consistency
        axes = [[axes[0], None], [axes[1], None]]
    
    num_epochs = len(history['train_loss'])
    epochs = range(1, num_epochs + 1)
    
    # Loss plot
    axes[0][0].plot(epochs, history['train_loss'], label='Train Loss', alpha=0.8, linewidth=1.5)
    axes[0][0].plot(epochs, history['val_loss'], label='Val Loss', alpha=0.8, linewidth=1.5)
    axes[0][0].set_xlabel('Epoch', fontsize=11)
    axes[0][0].set_ylabel('Loss', fontsize=11)
    axes[0][0].set_title(f'Fold {fold} - Loss (Epochs: 1-{num_epochs})', fontsize=12, fontweight='bold')
    axes[0][0].legend(loc='best', fontsize=10)
    axes[0][0].grid(True, alpha=0.3)
    # Ensure x-axis shows all epochs
    axes[0][0].set_xlim(1, num_epochs)
    if num_epochs > 20:
        # Show every 10th tick if many epochs
        axes[0][0].set_xticks(range(1, num_epochs + 1, max(1, num_epochs // 10)))
    
    # RMSE plot (if available)
    if has_rmse:
        rmse_epochs, rmse_values = zip(*history['val_rmse'])
        if history.get('train_rmse'):
            train_epochs, train_rmse_values = zip(*history['train_rmse'])
            axes[0][1].plot(train_epochs, train_rmse_values, label='Train RMSE', alpha=0.8, linewidth=1.5, marker='o', markersize=4)
        axes[0][1].plot(rmse_epochs, rmse_values, label='Val RMSE', alpha=0.8, linewidth=1.5, marker='s', markersize=4)
        axes[0][1].set_xlabel('Epoch', fontsize=11)
        axes[0][1].set_ylabel('RMSE (yards)', fontsize=11)
        axes[0][1].set_title(f'Fold {fold} - RMSE', fontsize=12, fontweight='bold')
        axes[0][1].legend(loc='best', fontsize=10)
        axes[0][1].grid(True, alpha=0.3)
        # Ensure x-axis shows all epochs
        axes[0][1].set_xlim(1, num_epochs)
    
    # Learning rate plot
    axes[1][0].plot(epochs, history['learning_rates'], alpha=0.8, linewidth=1.5, color='green')
    axes[1][0].set_xlabel('Epoch', fontsize=11)
    axes[1][0].set_ylabel('Learning Rate', fontsize=11)
    axes[1][0].set_title(f'Fold {fold} - Learning Rate Schedule', fontsize=12, fontweight='bold')
    axes[1][0].set_yscale('log')
    axes[1][0].grid(True, alpha=0.3)
    axes[1][0].set_xlim(1, num_epochs)
    
    # Loss difference (overfitting indicator) - use second column if RMSE exists, otherwise hide
    if has_rmse and len(history['train_loss']) == len(history['val_loss']):
        diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[1][1].plot(epochs, diff, alpha=0.8, color='orange', linewidth=1.5)
        axes[1][1].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No Overfitting')
        axes[1][1].fill_between(epochs, 0, diff, alpha=0.2, color='orange', where=(diff > 0))
        axes[1][1].set_xlabel('Epoch', fontsize=11)
        axes[1][1].set_ylabel('Val Loss - Train Loss', fontsize=11)
        axes[1][1].set_title(f'Fold {fold} - Overfitting Indicator', fontsize=12, fontweight='bold')
        axes[1][1].legend(loc='best', fontsize=10)
        axes[1][1].grid(True, alpha=0.3)
        axes[1][1].set_xlim(1, num_epochs)
    elif not has_rmse and len(history['train_loss']) == len(history['val_loss']):
        # Show overfitting indicator in second row if no RMSE plot
        diff = np.array(history['val_loss']) - np.array(history['train_loss'])
        axes[1][0].plot(epochs, diff, alpha=0.8, color='orange', linewidth=1.5, label='Overfitting Indicator')
        axes[1][0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1][0].fill_between(epochs, 0, diff, alpha=0.2, color='orange', where=(diff > 0))
        axes[1][0].set_xlabel('Epoch', fontsize=11)
        axes[1][0].set_ylabel('Val Loss - Train Loss', fontsize=11)
        axes[1][0].set_title(f'Fold {fold} - Overfitting Indicator', fontsize=12, fontweight='bold')
        axes[1][0].legend(loc='best', fontsize=10)
        axes[1][0].grid(True, alpha=0.3)
        axes[1][0].set_xlim(1, num_epochs)
    
    # Add overall title with training info
    fig.suptitle(f'Fold {fold} Training History ({num_epochs} epochs, Early Stop: {"Yes" if num_epochs < 200 else "No"})', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for suptitle
    plot_filename = f"fold_{fold}_history_{timestamp}.png" if timestamp else f"fold_{fold}_history.png"
    plt.savefig(output_dir / plot_filename, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_cv_summary(results: Dict, output_dir: Path, timestamp: str = ""):
    """Plot summary across all CV folds."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss across folds
    folds = range(1, len(results['fold_losses']) + 1)
    axes[0].bar(folds, results['fold_losses'], alpha=0.7)
    axes[0].axhline(y=results['mean_loss'], color='r', linestyle='--', 
                    label=f'Mean: {results["mean_loss"]:.4f}')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Validation Loss Across Folds')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # RMSE across folds
    axes[1].bar(folds, results['fold_rmses'], alpha=0.7, color='green')
    axes[1].axhline(y=results['mean_rmse'], color='r', linestyle='--',
                    label=f'Mean: {results["mean_rmse"]:.4f}')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Validation RMSE (yards)')
    axes[1].set_title('Validation RMSE Across Folds')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_filename = f"cv_summary_{timestamp}.png" if timestamp else "cv_summary.png"
    plt.savefig(output_dir / plot_filename, dpi=150)
    plt.close()

