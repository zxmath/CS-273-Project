"""
Main training script for NFL trajectory prediction models.

This script orchestrates the complete training pipeline:
1. Load training data
2. Prepare features and sequences
3. Train models with cross-validation
4. Save models and artifacts with timestamps

Usage:
    python scripts/train.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Fix OpenBLAS threading issues (set before importing numpy)
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pickle
import torch
from configs.Config import Config
from src.nfl_bdb_2026.io import load_training_data
from src.nfl_bdb_2026.features import prepare_sequences_geometric
from src.nfl_bdb_2026.model import STTransformer
from src.nfl_bdb_2026.cross_validation import cross_validate
from src.nfl_bdb_2026.util import set_seed


def get_timestamp() -> str:
    """Get current timestamp in YYYYMMDD_HHMMSS format."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def train_sttransformer(
    config: Config,
    sequences: list,
    targets_dx: list,
    targets_dy: list,
    groups: np.ndarray,
    route_kmeans,
    route_scaler,
    timestamp: str,
) -> dict:
    """
    Train STTransformer model with cross-validation.
    
    Args:
        config: Configuration object
        sequences: List of input sequences
        targets_dx: List of target dx arrays
        targets_dy: List of target dy arrays
        groups: Group IDs for GroupKFold (game_id)
        route_kmeans: Pre-fitted route clustering model
        route_scaler: Pre-fitted route scaler
        timestamp: Timestamp string for organizing outputs
        
    Returns:
        Dictionary with training results and artifacts
    """
    print(f"\n{'='*80}")
    print(f"Training STTransformer Model")
    print(f"{'='*80}")
    
    # Create separate timestamped directories for models and results
    model_dir = config.MODEL_DIR / f"sttransformer_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = config.RESULTS_DIR / f"sttransformer_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Model factory function
    input_dim = sequences[0].shape[-1]
    
    def model_factory():
        return STTransformer(
            input_dim=input_dim,
            hidden_dim=config.HIDDEN_DIM,
            horizon=config.MAX_FUTURE_HORIZON,
            window_size=config.WINDOW_SIZE,
            n_heads=config.STTRANSFORMER_N_HEADS,
            n_layers=config.STTRANSFORMER_N_LAYERS
        )
    
    print(f"  - Model configuration:")
    print(f"   Input dimension: {input_dim}")
    print(f"   Hidden dimension: {config.HIDDEN_DIM}")
    print(f"   Horizon: {config.MAX_FUTURE_HORIZON}")
    print(f"   Window size: {config.WINDOW_SIZE}")
    print(f"   Transformer heads: {config.STTRANSFORMER_N_HEADS}")
    print(f"   Transformer layers: {config.STTRANSFORMER_N_LAYERS}")
    print(f"   Models directory: {model_dir}")
    print(f"   Results directory: {results_dir}")
    
    # Run cross-validation (logs and plots go to results_dir)
    print(f"\n - Starting cross-validation... Models will be saved after all folds complete.")
    try:
        results = cross_validate(
            model_factory=model_factory,
            sequences=sequences,
            targets_dx=targets_dx,
            targets_dy=targets_dy,
            groups=groups,
            config=config,
            enable_logging=True,
            enable_plots=True,
            output_dir=results_dir,  # Results (logs/plots) go here
            log_interval=10,
        )
        print(f"\n - Cross-validation completed! Now saving models...")
    except Exception as e:
        print(f"  - Error during cross-validation: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Save route artifacts to model directory
    try:
        with open(model_dir / "route_kmeans.pkl", "wb") as f:
            pickle.dump(route_kmeans, f)
        with open(model_dir / "route_scaler.pkl", "wb") as f:
            pickle.dump(route_scaler, f)
        print(f"  - Saved route artifacts to {model_dir}")
    except Exception as e:
        print(f"  - Warning: Failed to save route artifacts: {e}")
    
    # Save models and scalers to model directory
    print(f"\n - Saving models and scalers to {model_dir}...")
    try:
        if 'models' not in results or 'scalers' not in results:
            print(f"  - Error: Results dictionary missing 'models' or 'scalers' keys")
            print(f"   Available keys: {list(results.keys())}")
            return None
        
        if len(results['models']) == 0:
            print(f"  - Error: No models to save!")
            return None
        
        if len(results['models']) != len(results['scalers']):
            print(f"   - Warning: Mismatch in number of models ({len(results['models'])}) and scalers ({len(results['scalers'])})")
        
        for i, (model, scaler) in enumerate(zip(results['models'], results['scalers']), 1):
            try:
                # Save model state
                model_path = model_dir / f"model_fold{i}.pt"
                state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(state, str(model_path))
                
                # Save scaler
                scaler_path = model_dir / f"scaler_fold{i}.pkl"
                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
                
                print(f"   - Fold {i}: {model_path.name}, {scaler_path.name}")
            except Exception as e:
                print(f"   - Error saving fold {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Verify files were actually saved
        saved_models = list(model_dir.glob("model_fold*.pt"))
        saved_scalers = list(model_dir.glob("scaler_fold*.pkl"))
        print(f"\n   - Successfully saved {len(saved_models)} model files and {len(saved_scalers)} scaler files to {model_dir}")
    except Exception as e:
        print(f"   - Error during model saving: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Save metadata to results directory
    metadata = {
        'timestamp': timestamp,
        'model_type': 'STTransformer',
        'input_dim': input_dim,
        'config': {
            # Training hyperparameters
            'SEED': config.SEED,
            'BATCH_SIZE': config.BATCH_SIZE,
            'EPOCHS': config.EPOCHS,
            'PATIENCE': config.PATIENCE,
            'LEARNING_RATE': config.LEARNING_RATE,
            'N_FOLDS': config.N_FOLDS,
            # Data hyperparameters
            'WINDOW_SIZE': config.WINDOW_SIZE,
            'HIDDEN_DIM': config.HIDDEN_DIM,
            'MAX_FUTURE_HORIZON': config.MAX_FUTURE_HORIZON,
            # STTransformer-specific hyperparameters
            'STTRANSFORMER_N_HEADS': config.STTRANSFORMER_N_HEADS,
            'STTRANSFORMER_N_LAYERS': config.STTRANSFORMER_N_LAYERS,
            'STTRANSFORMER_MLP_HIDDEN_DIM': config.STTRANSFORMER_MLP_HIDDEN_DIM,
            'STTRANSFORMER_N_RES_BLOCKS': config.STTRANSFORMER_N_RES_BLOCKS,
            # Autoencoder settings
            'STTRANSFORMER_USE_AE': config.STTRANSFORMER_USE_AE,
            'STTRANSFORMER_AE_LATENT_DIM': config.STTRANSFORMER_AE_LATENT_DIM,
            'STTRANSFORMER_AE_N_LAYERS': config.STTRANSFORMER_AE_N_LAYERS,
        },
        'cv_results': {
            'mean_loss': float(results['mean_loss']),
            'std_loss': float(results['std_loss']),
            'mean_rmse': float(results['mean_rmse']),
            'std_rmse': float(results['std_rmse']),
            'fold_losses': [float(x) for x in results['fold_losses']],
            'fold_rmses': [float(x) for x in results['fold_rmses']],
        },
        'model_dir': str(model_dir),  # Reference to where models are saved
    }
    
    with open(results_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"\n - STTransformer training complete!")
    print(f"   Mean RMSE: {results['mean_rmse']:.5f} ± {results['std_rmse']:.5f}")
    print(f"   Models saved to: {model_dir}")
    print(f"   Results saved to: {results_dir}")
    
    return {
        'model_type': 'STTransformer',
        'model_dir': model_dir,
        'results_dir': results_dir,
        'results': results,
        'metadata': metadata,
    }


def train_sttransformer_decoder(
    config: Config,
    sequences: list,
    targets_dx: list,
    targets_dy: list,
    groups: np.ndarray,
    route_kmeans,
    route_scaler,
    timestamp: str,
) -> dict:
    """
    Train STTransformerDecoder (encoder-decoder) model with cross-validation.
    
    Args:
        config: Configuration object
        sequences: List of input sequences
        targets_dx: List of target dx arrays
        targets_dy: List of target dy arrays
        groups: Group IDs for GroupKFold (game_id)
        route_kmeans: Pre-fitted route clustering model
        route_scaler: Pre-fitted route scaler
        timestamp: Timestamp string for organizing outputs
        
    Returns:
        Dictionary with training results and artifacts
    """
    print(f"\n{'='*80}")
    print(f" - Training STTransformerDecoder Model (Encoder-Decoder)")
    print(f"{'='*80}")
    
    # Create separate timestamped directories for models and results
    model_dir = config.MODEL_DIR / f"sttransformer_decoder_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = config.RESULTS_DIR / f"sttransformer_decoder_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Model factory function
    input_dim = sequences[0].shape[-1]
    
    from src.nfl_bdb_2026.models import get_model_factory
    
    def model_factory():
        factory_func = get_model_factory('sttransformer_decoder')
        if factory_func is None:
            raise ValueError("sttransformer_decoder model not found in registry")
        return factory_func(config=config, input_dim=input_dim)
    
    print(f"  - Model configuration:")
    print(f"   Input dimension: {input_dim}")
    print(f"   Hidden dimension: {config.HIDDEN_DIM}")
    print(f"   Horizon: {config.MAX_FUTURE_HORIZON}")
    print(f"   Window size: {config.WINDOW_SIZE}")
    decoder_n_heads = getattr(config, 'STTRANSFORMER_DECODER_N_HEADS', config.STTRANSFORMER_N_HEADS)
    decoder_n_layers = getattr(config, 'STTRANSFORMER_DECODER_N_LAYERS', config.STTRANSFORMER_N_LAYERS)
    print(f"   Transformer heads: {decoder_n_heads}")
    print(f"   Transformer layers: {decoder_n_layers}")
    use_ae = getattr(config, 'STTRANSFORMER_DECODER_USE_AE', False)
    print(f"   Autoencoder: {use_ae}")
    print(f"   Models directory: {model_dir}")
    print(f"   Results directory: {results_dir}")
    
    # Run cross-validation (logs and plots go to results_dir)
    try:
        results = cross_validate(
            model_factory=model_factory,
            sequences=sequences,
            targets_dx=targets_dx,
            targets_dy=targets_dy,
            groups=groups,
            config=config,
            enable_logging=True,
            enable_plots=True,
            output_dir=results_dir,
            log_interval=10,
        )
    except Exception as e:
        print(f"  - Error during cross-validation: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Save route artifacts to model directory
    try:
        with open(model_dir / "route_kmeans.pkl", "wb") as f:
            pickle.dump(route_kmeans, f)
        with open(model_dir / "route_scaler.pkl", "wb") as f:
            pickle.dump(route_scaler, f)
        print(f"   - Saved route artifacts to {model_dir}")
    except Exception as e:
        print(f"   - Warning: Failed to save route artifacts: {e}")
    
    # Save models and scalers to model directory
    print(f"\n - Saving models and scalers to {model_dir}...")
    try:
        if 'models' not in results or 'scalers' not in results:
            print(f"  - Error: Results dictionary missing 'models' or 'scalers' keys")
            print(f"   Available keys: {list(results.keys())}")
            return None
        
        if len(results['models']) == 0:
            print(f"  - Error: No models to save!")
            return None
        
        # Filter out None models (failed folds) and save only successful ones
        saved_count = 0
        for i, (model, scaler) in enumerate(zip(results['models'], results['scalers']), 1):
            # Skip failed folds (None models)
            if model is None:
                print(f"   - Fold {i}: Skipping (training failed)")
                continue
            
            try:
                # Save model state
                model_path = model_dir / f"model_fold{i}.pt"
                state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(state, str(model_path))
                
                # Save scaler (if available)
                if scaler is not None:
                    scaler_path = model_dir / f"scaler_fold{i}.pkl"
                    with open(scaler_path, "wb") as f:
                        pickle.dump(scaler, f)
                    print(f"   - Fold {i}: {model_path.name}, {scaler_path.name}")
                else:
                    print(f"   - Fold {i}: {model_path.name} (no scaler)")
                
                saved_count += 1
            except Exception as e:
                print(f"   - Error saving fold {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if saved_count == 0:
            print(f"   - Error: No models were successfully saved!")
            print(f"   All folds failed during training. Check training logs for errors.")
            return None
        
        print(f"   - Successfully saved {saved_count}/{len(results['models'])} model files")
        
        # Warn if some folds failed
        failed_count = len(results['models']) - saved_count
        if failed_count > 0:
            print(f"   - Warning: {failed_count} fold(s) failed during training and were not saved")
    
    except Exception as e:
        print(f"   - Error during model saving: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': 'STTransformerDecoder',
        'input_dim': input_dim,
        'config': {
            'HIDDEN_DIM': config.HIDDEN_DIM,
            'MAX_FUTURE_HORIZON': config.MAX_FUTURE_HORIZON,
            'WINDOW_SIZE': config.WINDOW_SIZE,
            'STTRANSFORMER_DECODER_N_HEADS': decoder_n_heads,
            'STTRANSFORMER_DECODER_N_LAYERS': decoder_n_layers,
            'STTRANSFORMER_DECODER_USE_TF': getattr(config, 'STTRANSFORMER_DECODER_USE_TF', True),
            'STTRANSFORMER_DECODER_USE_AE': use_ae,
            'BATCH_SIZE': config.BATCH_SIZE,
            'EPOCHS': config.EPOCHS,
            'LEARNING_RATE': config.LEARNING_RATE,
            'N_FOLDS': config.N_FOLDS,
        },
        'cv_results': {
            'mean_loss': float(results['mean_loss']),
            'std_loss': float(results['std_loss']),
            'mean_rmse': float(results['mean_rmse']),
            'std_rmse': float(results['std_rmse']),
            'fold_losses': [float(x) for x in results['fold_losses']],
            'fold_rmses': [float(x) for x in results['fold_rmses']],
        },
        'model_dir': str(model_dir),
    }
    
    with open(results_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"\n - STTransformerDecoder training complete!")
    print(f"   Mean RMSE: {results['mean_rmse']:.5f} ± {results['std_rmse']:.5f}")
    print(f"   Models saved to: {model_dir}")
    print(f"   Results saved to: {results_dir}")
    
    return {
        'model_type': 'STTransformerDecoder',
        'model_dir': model_dir,
        'results_dir': results_dir,
        'results': results,
        'metadata': metadata,
    }


def train_sklearn_model(
    model_name: str,
    config: Config,
    sequences: list,
    targets_dx: list,
    targets_dy: list,
    groups: np.ndarray,
    route_kmeans,
    route_scaler,
    timestamp: str,
    use_scaling: bool = True,
    model_kwargs: dict = None,
) -> dict:
    """
    Train sklearn-style model (linear, tree, ARIMA) with cross-validation.
    
    Args:
        model_name: Name of model ('ridge', 'lasso', 'xgboost', 'lightgbm', 'arima', etc.)
        config: Configuration object
        sequences: List of input sequences
        targets_dx: List of target dx arrays
        targets_dy: List of target dy arrays
        groups: Group IDs for GroupKFold
        route_kmeans: Pre-fitted route clustering model
        route_scaler: Pre-fitted route scaler
        timestamp: Timestamp string
        use_scaling: Whether to scale features (True for linear, False for trees)
        model_kwargs: Additional kwargs for model creation
        
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*80}")
    print(f" - Training {model_name.upper()} Model")
    print(f"{'='*80}")
    
    from src.nfl_bdb_2026.models import ModelFactory
    from src.nfl_bdb_2026.cross_validation_sklearn import cross_validate_sklearn
    
    # Create directories
    model_dir = config.MODEL_DIR / f"{model_name}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = config.RESULTS_DIR / f"{model_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Model factory
    input_dim = sequences[0].shape[-1]
    factory = ModelFactory(model_name, config)
    
    if model_kwargs is None:
        model_kwargs = {}
    
    def model_factory():
        return factory.create(input_dim=input_dim, **model_kwargs)
    
    print(f"  - Model configuration:")
    print(f"   Model type: {model_name}")
    print(f"   Input dimension: {input_dim} (flattened: {input_dim * config.WINDOW_SIZE})")
    print(f"   Horizon: {config.MAX_FUTURE_HORIZON}")
    print(f"   Feature scaling: {use_scaling}")
    print(f"   Models directory: {model_dir}")
    print(f"   Results directory: {results_dir}")
    
    # Run cross-validation
    # Note: ARIMA doesn't use scaling
    flatten = True  # Always flatten for sklearn models
    if model_name == 'arima':
        use_scaling = False
        flatten = False  # ARIMA handles sequences differently
    
    results = cross_validate_sklearn(
        model_factory=model_factory,
        sequences=sequences,
        targets_dx=targets_dx,
        targets_dy=targets_dy,
        groups=groups,
        config=config,
        use_scaling=use_scaling,
        flatten=flatten,
        output_dir=results_dir,
    )
    
    # Save route artifacts
    with open(model_dir / "route_kmeans.pkl", "wb") as f:
        pickle.dump(route_kmeans, f)
    with open(model_dir / "route_scaler.pkl", "wb") as f:
        pickle.dump(route_scaler, f)
    
    # Save models and scalers
    print(f"\n - Saving models to {model_dir}...")
    for i, (model, scaler) in enumerate(zip(results['models'], results['scalers']), 1):
        # Save model
        import joblib
        model_path = model_dir / f"model_fold{i}.pkl"
        joblib.dump(model, str(model_path))
        
        # Save scaler (if exists)
        if scaler is not None:
            scaler_path = model_dir / f"scaler_fold{i}.pkl"
            joblib.dump(scaler, str(scaler_path))
            print(f"   - Fold {i}: {model_path.name}, {scaler_path.name}")
        else:
            print(f"   - Fold {i}: {model_path.name}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_type': model_name,
        'input_dim': input_dim,
        'config': {
            'MAX_FUTURE_HORIZON': config.MAX_FUTURE_HORIZON,
            'WINDOW_SIZE': config.WINDOW_SIZE,
            'N_FOLDS': config.N_FOLDS,
            'use_scaling': use_scaling,
            'model_kwargs': model_kwargs if model_kwargs else {},
        },
        'cv_results': {
            'mean_train_loss': float(results.get('mean_train_loss', 0)),
            'std_train_loss': float(results.get('std_train_loss', 0)),
            'mean_loss': float(results['mean_loss']),
            'std_loss': float(results['std_loss']),
            'mean_rmse': float(results['mean_rmse']),
            'std_rmse': float(results['std_rmse']),
            'fold_train_losses': [float(x) for x in results.get('fold_train_losses', [])],
            'fold_losses': [float(x) for x in results['fold_losses']],
            'fold_rmses': [float(x) for x in results['fold_rmses']],
        },
        'model_dir': str(model_dir),
    }
    
    with open(results_dir / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    print(f"\n - {model_name.upper()} training complete!")
    if 'mean_train_loss' in results:
        print(f"   Mean Train Loss: {results['mean_train_loss']:.5f} ± {results['std_train_loss']:.5f}")
        print(f"   Mean Val Loss: {results['mean_loss']:.5f} ± {results['std_loss']:.5f}")
    print(f"   Mean Val RMSE: {results['mean_rmse']:.5f} ± {results['std_rmse']:.5f}")
    print(f"   Models saved to: {model_dir}")
    print(f"   Results saved to: {results_dir}")
    
    return {
        'model_type': model_name,
        'model_dir': model_dir,
        'results_dir': results_dir,
        'results': results,
        'metadata': metadata,
    }


def main():
    """Main training pipeline."""
    print("="*80)
    print(" - NFL BIG DATA BOWL 2026 - TRAINING PIPELINE")
    print("="*80)
    
    # Initialize
    config = Config()
    timestamp = get_timestamp()
    set_seed(config.SEED)
    
    print(f"\n - Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" - Timestamp: {timestamp}")
    print(f" - Device: {config.DEVICE}")
    print(f" - Data Directory: {config.DATA_DIR}")
    print(f" - Models Directory: {config.MODEL_DIR}")
    print(f" - Results Directory: {config.RESULTS_DIR}")
    print(f" - Cross-Validation Folds: {config.N_FOLDS}")
    print(f" - Debug Mode: {config.DEBUG}")
    
    # ========================================================================
    # STEP 1: LOAD TRAINING DATA
    # ========================================================================
    print(f"\n{'='*80}")
    print("[1/4] - Loading training data...")
    print(f"{'='*80}")
    
    train_input, train_output = load_training_data(
        debug=config.DEBUG,
        sample_plays=100 if config.DEBUG else None
    )
    
    print(f" - Loaded {len(train_input):,} input rows")
    print(f" - Loaded {len(train_output):,} output rows")
    
    # ========================================================================
    # STEP 2: PREPARE SEQUENCES WITH FEATURES
    # ========================================================================
    print(f"\n{'='*80}")
    print("[2/4] - Preparing sequences with geometric features...")
    print(f"{'='*80}")
    
    result = prepare_sequences_geometric(
        train_input,
        train_output,
        is_training=True,
        window_size=config.WINDOW_SIZE
    )
    
    (sequences, targets_dx, targets_dy, targets_frame_ids, sequence_ids,
     geo_endpoints_x, geo_endpoints_y, route_kmeans, route_scaler) = result
    
    sequences = list(sequences)
    targets_dx = list(targets_dx)
    targets_dy = list(targets_dy)
    
    print(f" - Created {len(sequences):,} training sequences")
    if sequences:
        print(f" - Feature dimension: {sequences[0].shape}")
    
    # ========================================================================
    # STEP 3: TRAIN MODELS
    # ========================================================================
    print(f"\n{'='*80}")
    print("[3/4] - Training models...")
    print(f"{'='*80}")
    
    # Prepare groups for GroupKFold
    groups = np.array([d['game_id'] for d in sequence_ids])
    
    # Train multiple models
    training_results = []
    '''
    # 1. Train STTransformer (neural network)
    stt_result = train_sttransformer(
       config=config,
       sequences=sequences,
       targets_dx=targets_dx,
       targets_dy=targets_dy,
       groups=groups,
       route_kmeans=route_kmeans,
       route_scaler=route_scaler,
       timestamp=timestamp,
    )
    if stt_result:
       training_results.append(stt_result)
    '''
    # 1b. Train STTransformerDecoder (encoder-decoder with autoregressive decoder)
    try:
        stt_decoder_result = train_sttransformer_decoder(
            config=config,
            sequences=sequences,
            targets_dx=targets_dx,
            targets_dy=targets_dy,
            groups=groups,
            route_kmeans=route_kmeans,
            route_scaler=route_scaler,
            timestamp=timestamp,
        )
        if stt_decoder_result:
            training_results.append(stt_decoder_result)
    except Exception as e:
        print(f"   - STTransformerDecoder training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Train Linear Models
    #print(f"\n{'='*80}")
    #print("📋 Training Linear Models...")
    #print(f"{'='*80}")
    
    
    '''
    # Ridge Regression
    try:
        ridge_result = train_sklearn_model(
            model_name='ridge',
            config=config,
            sequences=sequences,
            targets_dx=targets_dx,
            targets_dy=targets_dy,
            groups=groups,
            route_kmeans=route_kmeans,
            route_scaler=route_scaler,
            timestamp=timestamp,
            use_scaling=True,
            model_kwargs={'alpha': config.RIDGE_ALPHA},
        )
        training_results.append(ridge_result)
    except Exception as e:
        print(f"   - Ridge training failed: {e}")
    
    # Lasso Regression
    try:
        lasso_result = train_sklearn_model(
            model_name='lasso',
            config=config,
            sequences=sequences,
            targets_dx=targets_dx,
            targets_dy=targets_dy,
            groups=groups,
            route_kmeans=route_kmeans,
            route_scaler=route_scaler,
            timestamp=timestamp,
            use_scaling=True,
            model_kwargs={'alpha': config.LASSO_ALPHA},
        )
        training_results.append(lasso_result)
    except Exception as e:
        print(f"   - Lasso training failed: {e}")
    
    # ElasticNet
    try:
        elasticnet_result = train_sklearn_model(
            model_name='elasticnet',
            config=config,
            sequences=sequences,
            targets_dx=targets_dx,
            targets_dy=targets_dy,
            groups=groups,
            route_kmeans=route_kmeans,
            route_scaler=route_scaler,
            timestamp=timestamp,
            use_scaling=True,
            model_kwargs={'alpha': config.ELASTICNET_ALPHA, 'l1_ratio': config.ELASTICNET_L1_RATIO},
        )
        training_results.append(elasticnet_result)
    except Exception as e:
        print(f"   - ElasticNet training failed: {e}")
    '''
    '''
    # 3. Train Tree Models
    print(f"\n{'='*80}")
    print(" - Training Tree Models...")
    print(f"{'='*80}")
    
    # XGBoost
    try:
        xgb_result = train_sklearn_model(
            model_name='xgboost',
            config=config,
            sequences=sequences,
            targets_dx=targets_dx,
            targets_dy=targets_dy,
            groups=groups,
            route_kmeans=route_kmeans,
            route_scaler=route_scaler,
            timestamp=timestamp,
            use_scaling=False,  # Trees don't need scaling
            model_kwargs={
                'n_estimators': config.TREE_N_ESTIMATORS,
                'max_depth': config.TREE_MAX_DEPTH,
                'learning_rate': config.TREE_LEARNING_RATE
            },
        )
        training_results.append(xgb_result)
    except ImportError as e:
        print(f"   - XGBoost not available: {e}")
        print("  Install with: pip install xgboost")
    except Exception as e:
        print(f"   - XGBoost training failed: {e}")
    
    # LightGBM
    try:
        lgb_result = train_sklearn_model(
            model_name='lightgbm',
            config=config,
            sequences=sequences,
            targets_dx=targets_dx,
            targets_dy=targets_dy,
            groups=groups,
            route_kmeans=route_kmeans,
            route_scaler=route_scaler,
            timestamp=timestamp,
            use_scaling=False,
            model_kwargs={
                'n_estimators': config.TREE_N_ESTIMATORS,
                'max_depth': config.TREE_MAX_DEPTH,
                'learning_rate': config.TREE_LEARNING_RATE
            },
        )
        training_results.append(lgb_result)
    except ImportError as e:
        print(f"   - LightGBM not available: {e}")
        print("  Install with: pip install lightgbm")
    except Exception as e:
        print(f"   - LightGBM training failed: {e}")
    
    # 4. Train ARIMA (experimental)
    print(f"\n{'='*80}")
    print(" - Training ARIMA Model (experimental)...")
    print(f"{'='*80}")
    
    try:
        arima_result = train_sklearn_model(
            model_name='arima',
            config=config,
            sequences=sequences,
            targets_dx=targets_dx,
            targets_dy=targets_dy,
            groups=groups,
            route_kmeans=route_kmeans,
            route_scaler=route_scaler,
            timestamp=timestamp,
            use_scaling=False,
            model_kwargs={'order': config.ARIMA_ORDER},
        )
        training_results.append(arima_result)
    except ImportError as e:
        print(f"   - ARIMA not available: {e}")
        print("  Install with: pip install statsmodels")
    except Exception as e:
        print(f"   - ARIMA training failed: {e}")
    '''
    # ========================================================================
    # STEP 4: FINAL SUMMARY
    # ========================================================================
    print(f"\n{'='*80}")
    print("[4/4] - Training Summary")
    print(f"{'='*80}")
    
    print(f"\n - Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n - Model Performance:")
    
    for result in training_results:
        print(f"\n  {result['model_type']}:")
        print(f"    Mean RMSE: {result['results']['mean_rmse']:.5f} ± {result['results']['std_rmse']:.5f}")
        print(f"    Mean Loss: {result['results']['mean_loss']:.5f} ± {result['results']['std_loss']:.5f}")
        print(f"    Models Directory: {result['model_dir']}")
        print(f"    Results Directory: {result['results_dir']}")
    
    print(f"\n{'='*80}")
    print(" - Training pipeline complete!")
    print(f"{'='*80}\n")
    
    return training_results


if __name__ == "__main__":
    try:
        results = main()
        print("\n - Success! All models trained successfully.")
    except Exception as e:
        print(f"\n - Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
