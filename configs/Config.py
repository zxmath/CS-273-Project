from pathlib import Path
import torch
class Config:
    # Set DATA_DIR to actual data location
    DATA_DIR = Path("")
    OUTPUT_DIR = Path("./outputs")
    OUTPUT_DIR.mkdir(exist_ok=True)
    # Separate directories for models and results
    MODEL_DIR = OUTPUT_DIR / "models"  # Save trained models and scalers here
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR = OUTPUT_DIR / "results"  # Save logs, plots, metadata here
    RESULTS_DIR.mkdir(exist_ok=True)
    # Toggle saving/loading of artifacts
    SAVE_ARTIFACTS = True
    LOAD_ARTIFACTS = False  # Set to False to force training
    LOAD_DIR = None  # No pre-trained models to load
  
    SEED = 42
    N_FOLDS = 5
    BATCH_SIZE = 512
    EPOCHS = 300
    PATIENCE = 50
    LEARNING_RATE = 1e-3
    
    WINDOW_SIZE = 10
    HIDDEN_DIM = 256
    MAX_FUTURE_HORIZON = 94
    
    # === STTransformer-specific hyperparameters ===
    STTRANSFORMER_N_HEADS = 8  # Number of attention heads
    STTRANSFORMER_N_LAYERS = 4  # Number of transformer encoder layers
    STTRANSFORMER_MLP_HIDDEN_DIM = 256  # Hidden dimension for ResidualMLP head
    STTRANSFORMER_N_RES_BLOCKS = 2  # Number of residual blocks in MLP head
    STTRANSFORMER_USE_AE = True  # Use autoencoder for feature compression
    STTRANSFORMER_AE_LATENT_DIM = None  # Autoencoder latent dim (None = input_dim // 2)
    STTRANSFORMER_AE_N_LAYERS = 2  # Number of autoencoder layers
    # ================================================
    
    # === STTransformerDecoder (Encoder-Decoder) hyperparameters ===
    STTRANSFORMER_DECODER_N_HEADS = 4  # Number of attention heads
    STTRANSFORMER_DECODER_N_LAYERS = 4  # Total layers (split between encoder/decoder)
    STTRANSFORMER_DECODER_USE_TF = True  # Use teacher forcing during training
    STTRANSFORMER_DECODER_USE_SCHEDULED_SAMPLING = True  # Use scheduled sampling to bridge train/val gap
    STTRANSFORMER_DECODER_SS_START_PROB = 1.0  # Start with 100% teacher forcing
    STTRANSFORMER_DECODER_SS_END_PROB = 0.0  # End with 0% teacher forcing (fully autoregressive)
    STTRANSFORMER_DECODER_SS_DECAY_EPOCHS = 50  # Decay over 50 epochs (or half of total epochs if less)
    STTRANSFORMER_DECODER_USE_AE = True  # Use autoencoder for feature compression
    STTRANSFORMER_DECODER_AE_LATENT_DIM = None # Autoencoder latent dim (None = input_dim // 2)
    STTRANSFORMER_DECODER_AE_N_LAYERS = 2  # Number of autoencoder layers
    # ===============================================================
    
    # === Linear Models hyperparameters ===
    RIDGE_ALPHA = 1.0  # Ridge regularization strength
    LASSO_ALPHA = 0.1  # Lasso regularization strength
    ELASTICNET_ALPHA = 0.1  # ElasticNet regularization strength
    ELASTICNET_L1_RATIO = 0.5  # ElasticNet L1/L2 mixing (0.5 = balanced)
    # =====================================
    
    # === Tree Models hyperparameters ===
    TREE_N_ESTIMATORS = 500  # Number of boosting rounds
    TREE_MAX_DEPTH = 8  # Maximum tree depth
    TREE_LEARNING_RATE = 0.05  # Learning rate for boosting
    TREE_SUBSAMPLE = 0.8  # Subsample ratio (for XGBoost/LightGBM) - regularization
    TREE_COLSAMPLE_BYTREE = 0.8  # Column subsample ratio - regularization
    TREE_MIN_CHILD_WEIGHT = 5  # Minimum samples per leaf (prevents overfitting)
    TREE_REG_ALPHA = 0.1  # L1 regularization strength
    TREE_REG_LAMBDA = 1.0  # L2 regularization strength
    TREE_EARLY_STOPPING_ROUNDS = 0  # Early stopping rounds (0 = disabled)
    # ====================================
    
    # === ARIMA hyperparameters ===
    ARIMA_ORDER = (2, 1, 2)  # ARIMA(p, d, q) order: (AR, differencing, MA)
    # =============================
    
    FIELD_X_MIN, FIELD_X_MAX = 0.0, 120.0
    FIELD_Y_MIN, FIELD_Y_MAX = 0.0, 53.3
    
    K_NEIGH = 6
    RADIUS = 30.0
    TAU = 8.0
    N_ROUTE_CLUSTERS = 7
    GPU_ID = 5
    DEVICE = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    DEBUG = False # Set to True for quick testing with less data
    if DEBUG:
        N_FOLDS = 5  # Reduce folds for debugging
