"""
Quick script to view cross-validation results from any model's metadata.pkl file.

Usage:
    python scripts/view_results.py outputs/results/xgboost_20251129_030247
    python scripts/view_results.py outputs/results/lightgbm_20251129_030247
"""

import sys
import pickle
import json
from pathlib import Path


def view_cv_results(results_dir):
    """View cross-validation results from metadata.pkl"""
    results_path = Path(results_dir) / "metadata.pkl"
    
    if not results_path.exists():
        print(f" - Results file not found: {results_path}")
        return
    
    try:
        with open(results_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f" - Error loading results: {e}")
        return
    
    print(f"\n{'='*60}")
    print(f"Model: {data.get('model_type', 'Unknown')}")
    print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
    print(f"{'='*60}\n")
    
    cv_results = data.get('cv_results', {})
    if not cv_results:
        print(" - No cross-validation results found in metadata")
        return
    
    print(" - Cross-Validation Results:")
    print(f"  Mean RMSE: {cv_results.get('mean_rmse', 'N/A'):.5f} ± {cv_results.get('std_rmse', 0):.5f}")
    print(f"  Mean Loss: {cv_results.get('mean_loss', 'N/A'):.5f} ± {cv_results.get('std_loss', 0):.5f}")
    
    fold_losses = cv_results.get('fold_losses', [])
    fold_rmses = cv_results.get('fold_rmses', [])
    
    if fold_losses and fold_rmses:
        print(f"\n - Per-Fold Results:")
        for i, (loss, rmse) in enumerate(zip(fold_losses, fold_rmses), 1):
            print(f"  Fold {i}: Loss={loss:.5f}, RMSE={rmse:.5f}")
    
    # Check for additional files
    results_dir_path = Path(results_dir)
    log_files = list(results_dir_path.glob("*.log"))
    plot_files = list(results_dir_path.glob("*.png"))
    
    if log_files or plot_files:
        print(f"\n - Additional Files:")
        for log_file in log_files:
            print(f"   - {log_file.name}")
        for plot_file in plot_files:
            print(f"   - {plot_file.name}")
    
    print(f"{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print(" - Usage: python scripts/view_results.py <results_directory>")
        print("\nExample:")
        print("  python scripts/view_results.py outputs/results/xgboost_20251129_030247")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    view_cv_results(results_dir)


if __name__ == "__main__":
    main()

