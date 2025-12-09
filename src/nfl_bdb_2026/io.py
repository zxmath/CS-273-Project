"""
Data I/O utilities for loading NFL training data.

Functions:
- load_training_data: Load input and output CSV files
"""

import os
# Fix OpenBLAS threading issues (set before importing numpy/pandas)
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from tqdm.auto import tqdm

import sys
from pathlib import Path as PathType

# Add project root to sys.path for Config import
PROJECT_ROOT = PathType(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from configs.Config import Config


def load_training_data(
    data_dir: Optional[Path] = None,
    debug: bool = False,
    max_weeks: Optional[int] = None,
    sample_plays: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training input and output data from CSV files.
    
    Args:
        data_dir: Directory containing train/ folder (defaults to Config.DATA_DIR)
        debug: If True, load only a subset of data for quick testing
        max_weeks: Maximum number of weeks to load (1-18, None = all)
        sample_plays: If provided, sample this many plays from first week (for debug)
        
    Returns:
        train_input: DataFrame with input tracking data
        train_output: DataFrame with output ground truth trajectories
        
    Example:
        # Load all data
        train_input, train_output = load_training_data()
        
        # Load debug subset
        train_input, train_output = load_training_data(debug=True)
        
        # Load first 3 weeks
        train_input, train_output = load_training_data(max_weeks=3)
    """
    if data_dir is None:
        data_dir = Config.DATA_DIR
    
    train_dir = data_dir / "train"
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {train_dir}")
    
    # Determine which weeks to load
    if debug:
        weeks = [1]  # Only load week 1 in debug mode
    elif max_weeks is not None:
        weeks = list(range(1, min(max_weeks + 1, 19)))
    else:
        weeks = list(range(1, 19))  # Load all 18 weeks
    
    print(f" - Loading training data from {len(weeks)} week(s)...")
    
    # Build file paths
    train_input_files = [train_dir / f"input_2023_w{w:02d}.csv" for w in weeks]
    train_output_files = [train_dir / f"output_2023_w{w:02d}.csv" for w in weeks]
    
    # Filter to existing files
    train_input_files = [f for f in train_input_files if f.exists()]
    train_output_files = [f for f in train_output_files if f.exists()]
    
    if not train_input_files:
        raise FileNotFoundError(f"No input files found in {train_dir}")
    
    print(f"  Found {len(train_input_files)} input files and {len(train_output_files)} output files")
    
    # Load data
    train_input_dfs = []
    train_output_dfs = []
    
    for input_file, output_file in tqdm(
        zip(train_input_files, train_output_files),
        total=len(train_input_files),
        desc="Loading files"
    ):
        if input_file.exists():
            train_input_dfs.append(pd.read_csv(input_file))
        if output_file.exists():
            train_output_dfs.append(pd.read_csv(output_file))
    
    # Concatenate all data
    train_input = pd.concat(train_input_dfs, ignore_index=True) if train_input_dfs else pd.DataFrame()
    train_output = pd.concat(train_output_dfs, ignore_index=True) if train_output_dfs else pd.DataFrame()
    
    print(f"   - Loaded {len(train_input):,} input rows, {len(train_output):,} output rows")
    
    # Debug mode: sample plays if requested
    if debug and sample_plays:
        print(f"   - DEBUG MODE: Sampling {sample_plays} plays from first week...")
        # Get sample of plays from first loaded file
        sample_plays_df = train_output[['game_id', 'play_id']].drop_duplicates().head(sample_plays)
        train_input = train_input.merge(sample_plays_df, on=['game_id', 'play_id'], how='inner')
        train_output = train_output.merge(sample_plays_df, on=['game_id', 'play_id'], how='inner')
        print(f"   - Reduced to {len(sample_plays_df)} unique plays")
    
    # Filter by player_to_predict if column exists
    # Note: For training, we keep ALL players in input (for context/features),
    # but only predict players where player_to_predict==True
    # Output data already only contains players to predict
    if 'player_to_predict' in train_input.columns:
        num_total = len(train_input)
        num_to_predict = train_input['player_to_predict'].sum()
        print(f"   - Players: {num_to_predict:,} to predict out of {num_total:,} total rows")
    
    # Remove problematic plays if they exist (optional - can be removed if not needed)
    # train_input = train_input[(train_input.game_id != 2023091100) | (train_input.play_id != 3167)]
    # train_output = train_output[(train_output.game_id != 2023091100) | (train_output.play_id != 3167)]
    
    return train_input, train_output
