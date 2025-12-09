"""
Cross-validation for sklearn-style models (linear, tree, etc.).

These models use fit/predict interface and don't require iterative training.
"""

import numpy as np
from pathlib import Path
from typing import Callable, Optional, Dict, Any, List
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
import logging
import matplotlib.pyplot as plt
from datetime import datetime

from .training_sklearn import (
    prepare_data_for_sklearn,
    train_sklearn_model,
    compute_val_rmse_sklearn
)


def cross_validate_sklearn(
    model_factory: Callable[[], Any],
    sequences: List[np.ndarray],
    targets_dx: List[np.ndarray],
    targets_dy: List[np.ndarray],
    groups: np.ndarray,
    config: Any,
    use_scaling: bool = True,
    flatten: bool = True,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Perform cross-validation for sklearn-style models.
    
    Args:
        model_factory: Function that creates a new model instance
        sequences: List of input sequences (n_samples, window_size, features)
        targets_dx: List of target dx arrays (variable length)
        targets_dy: List of target dy arrays (variable length)
        groups: Group IDs for GroupKFold (game_id)
        config: Configuration object
        use_scaling: Whether to scale features (True for linear, False for trees)
        flatten: Whether to flatten sequences (True for sklearn models)
        output_dir: Directory for saving logs/results
        
    Returns:
        Dictionary with CV results
    """
    # Setup logging
    logger = None
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        log_file = output_dir / f"cross_validation_{timestamp_str}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.info(f"Starting {config.N_FOLDS}-fold cross-validation")
        logger.info(f"Total samples: {len(sequences)}")
    
    gkf = GroupKFold(n_splits=config.N_FOLDS)
    
    models = []
    scalers = []
    fold_losses = []
    fold_train_losses = []
    fold_rmses = []
    fold_histories = []
    
    for fold, (tr_idx, va_idx) in enumerate(tqdm(gkf.split(sequences, groups=groups), 
                                                   total=config.N_FOLDS, 
                                                   desc="CV Folds"), 1):
        if logger:
            logger.info(f"\n{'='*60}")
            logger.info(f"Fold {fold}/{config.N_FOLDS}")
            logger.info(f"{'='*60}")
            logger.info(f"Train samples: {len(tr_idx)}, Val samples: {len(va_idx)}")
        
        # Split data
        X_tr = [sequences[i] for i in tr_idx]
        X_va = [sequences[i] for i in va_idx]
        y_tr_dx = [targets_dx[i] for i in tr_idx]
        y_va_dx = [targets_dx[i] for i in va_idx]
        y_tr_dy = [targets_dy[i] for i in tr_idx]
        y_va_dy = [targets_dy[i] for i in va_idx]
        
        # Prepare data
        if flatten:
            X_tr_flat, y_tr_dx_pad, y_tr_dy_pad, masks_tr = prepare_data_for_sklearn(
                X_tr, y_tr_dx, y_tr_dy, config.MAX_FUTURE_HORIZON, flatten=True
            )
            X_va_flat, y_va_dx_pad, y_va_dy_pad, masks_va = prepare_data_for_sklearn(
                X_va, y_va_dx, y_va_dy, config.MAX_FUTURE_HORIZON, flatten=True
            )
        else:
            # For ARIMA: use sequences as stacked array, but prepare padded targets
            X_tr_flat = np.stack(X_tr)  # (n_samples, window_size, features)
            X_va_flat = np.stack(X_va)
            
            # Prepare padded targets
            y_tr_dx_padded = []
            y_tr_dy_padded = []
            y_va_dx_padded = []
            y_va_dy_padded = []
            masks_tr_list = []
            masks_va_list = []
            
            for dx, dy in zip(y_tr_dx, y_tr_dy):
                n_steps = len(dx)
                dx_pad = np.pad(dx, (0, config.MAX_FUTURE_HORIZON - n_steps), constant_values=0).astype(np.float32)
                dy_pad = np.pad(dy, (0, config.MAX_FUTURE_HORIZON - n_steps), constant_values=0).astype(np.float32)
                mask = np.zeros(config.MAX_FUTURE_HORIZON, dtype=np.float32)
                mask[:n_steps] = 1.0
                y_tr_dx_padded.append(dx_pad)
                y_tr_dy_padded.append(dy_pad)
                masks_tr_list.append(mask)
            
            for dx, dy in zip(y_va_dx, y_va_dy):
                n_steps = len(dx)
                dx_pad = np.pad(dx, (0, config.MAX_FUTURE_HORIZON - n_steps), constant_values=0).astype(np.float32)
                dy_pad = np.pad(dy, (0, config.MAX_FUTURE_HORIZON - n_steps), constant_values=0).astype(np.float32)
                mask = np.zeros(config.MAX_FUTURE_HORIZON, dtype=np.float32)
                mask[:n_steps] = 1.0
                y_va_dx_padded.append(dx_pad)
                y_va_dy_padded.append(dy_pad)
                masks_va_list.append(mask)
            
            y_tr_dx_pad = np.stack(y_tr_dx_padded)
            y_tr_dy_pad = np.stack(y_tr_dy_padded)
            y_va_dx_pad = np.stack(y_va_dx_padded)
            y_va_dy_pad = np.stack(y_va_dy_padded)
            masks_tr = np.stack(masks_tr_list)
            masks_va = np.stack(masks_va_list)
        
        # Create and train model
        if logger:
            logger.info("Creating and training model...")
        model = model_factory()
        
        trained_model, scaler, history = train_sklearn_model(
            model=model,
            X_train=X_tr_flat,
            y_train_dx=y_tr_dx_pad,
            y_train_dy=y_tr_dy_pad,
            X_val=X_va_flat,
            y_val_dx=y_va_dx_pad,
            y_val_dy=y_va_dy_pad,
            masks_val=masks_va,
            masks_train=masks_tr,
            scaler=None,
            use_scaling=use_scaling
        )
        
        # Compute validation RMSE on original variable-length targets
        if logger:
            logger.info("Computing validation RMSE...")
        val_rmse = compute_val_rmse_sklearn(
            trained_model,
            X_va,
            y_va_dx,
            y_va_dy,
            config.MAX_FUTURE_HORIZON,
            scaler=scaler,
            use_scaling=use_scaling
        )
        
        models.append(trained_model)
        scalers.append(scaler)
        train_loss = history['train_loss'][0]
        val_loss = history['val_loss'][0]
        fold_train_losses.append(train_loss)
        fold_losses.append(val_loss)
        fold_rmses.append(val_rmse)
        fold_histories.append(history)
        
        if logger:
            logger.info(f"Fold {fold} - Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}, Val RMSE: {val_rmse:.5f}")
        print(f"  Fold {fold}/{config.N_FOLDS}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}, Val RMSE={val_rmse:.5f}")
    
    # Aggregate results
    mean_train_loss = np.mean(fold_train_losses)
    std_train_loss = np.std(fold_train_losses)
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    mean_rmse = np.mean(fold_rmses)
    std_rmse = np.std(fold_rmses)
    
    if logger:
        logger.info(f"\n{'='*60}")
        logger.info("Cross-Validation Results")
        logger.info(f"{'='*60}")
        logger.info(f"Mean Train Loss: {mean_train_loss:.5f} ± {std_train_loss:.5f}")
        logger.info(f"Mean Val Loss: {mean_loss:.5f} ± {std_loss:.5f}")
        logger.info(f"Mean Val RMSE: {mean_rmse:.5f} ± {std_rmse:.5f}")
        logger.info(f"\nFold Results:")
        for i, (train_loss, val_loss, rmse) in enumerate(zip(fold_train_losses, fold_losses, fold_rmses), 1):
            logger.info(f"  Fold {i}: Train Loss={train_loss:.5f}, Val Loss={val_loss:.5f}, Val RMSE={rmse:.5f}")
    
    results = {
        'models': models,
        'scalers': scalers,
        'fold_train_losses': fold_train_losses,
        'fold_losses': fold_losses,
        'fold_rmses': fold_rmses,
        'mean_train_loss': mean_train_loss,
        'std_train_loss': std_train_loss,
        'mean_loss': mean_loss,
        'std_loss': std_loss,
        'mean_rmse': mean_rmse,
        'std_rmse': std_rmse,
        'fold_histories': fold_histories,
    }
    
    # Create plots if output_dir is provided
    if output_dir:
        _plot_cv_summary_sklearn(results, output_dir, timestamp_str)
    
    return results


def _plot_cv_summary_sklearn(results: Dict[str, Any], output_dir: Path, timestamp: str):
    """Plot cross-validation summary for sklearn models."""
    fold_losses = results['fold_losses']
    fold_rmses = results['fold_rmses']
    mean_loss = results['mean_loss']
    std_loss = results['std_loss']
    mean_rmse = results['mean_rmse']
    std_rmse = results['std_rmse']
    n_folds = len(fold_losses)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Fold-wise Loss plot
    folds = list(range(1, n_folds + 1))
    axes[0].bar(folds, fold_losses, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1.5)
    axes[0].axhline(y=mean_loss, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_loss:.4f} ± {std_loss:.4f}')
    axes[0].fill_between(range(0, n_folds + 2), 
                         mean_loss - std_loss, 
                         mean_loss + std_loss, 
                         alpha=0.2, color='red', label=f'±1 std')
    axes[0].set_xlabel('Fold', fontsize=12)
    axes[0].set_ylabel('Validation Loss', fontsize=12)
    axes[0].set_title('Cross-Validation Loss by Fold', fontsize=13, fontweight='bold')
    axes[0].set_xticks(folds)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].legend(loc='best', fontsize=10)
    
    # Fold-wise RMSE plot
    axes[1].bar(folds, fold_rmses, alpha=0.7, color='coral', edgecolor='black', linewidth=1.5)
    axes[1].axhline(y=mean_rmse, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rmse:.4f} ± {std_rmse:.4f}')
    axes[1].fill_between(range(0, n_folds + 2), 
                         mean_rmse - std_rmse, 
                         mean_rmse + std_rmse, 
                         alpha=0.2, color='red', label=f'±1 std')
    axes[1].set_xlabel('Fold', fontsize=12)
    axes[1].set_ylabel('Validation RMSE (yards)', fontsize=12)
    axes[1].set_title('Cross-Validation RMSE by Fold', fontsize=13, fontweight='bold')
    axes[1].set_xticks(folds)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].legend(loc='best', fontsize=10)
    
    # Add overall title
    fig.suptitle(f'Cross-Validation Summary ({n_folds} folds)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"cv_summary_{timestamp}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved CV summary plot: {plot_path.name}")

