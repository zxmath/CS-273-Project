# CS 273 Project — NFL Big Data Bowl 2026

This repository contains our CS 273 (Intro to Machine Learning) project by Xu Zhuang, Natanael Alpay, and Emeric Battaglia.

We participate in the Kaggle competition [NFL Big Data Bowl 2026 — Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction), which focuses on predicting player movement during frames when the ball is in the air. Our approach explores multiple model families, with emphasis on spatio‑temporal Transformers.

**Key components**
- `scripts/train.py`: Main training entrypoint with cross‑validation and artifact saving.
- `scripts/view_results.py`: Visualizes logs/plots and summarizes results.
- `configs/Config.py`: Central configuration of data paths and hyperparameters.
- `src/nfl_bdb_2026/`: Data IO, feature engineering, model definitions, training loops, CV utilities.

## Getting Started

### Prerequisites
- Python 3.10+ (managed via Poetry or system Python)
- macOS with `zsh` shell (commands below use `zsh`)
- GPU is optional; code falls back to CPU automatically.

### Installation
You can use Poetry (recommended) or plain `pip`.

Using Poetry:

```zsh
# From the repo root
brew install poetry  # if Poetry is not installed
poetry install
poetry run python --version
```

Using pip:

```zsh
# From the repo root
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e .
```

## Data Setup

Place the competition dataset under `Data/` (or wherever your local copy lives). Then set the absolute path in `configs/Config.py`:

```python
# configs/Config.py
from pathlib import Path
DATA_DIR = Path("/absolute/path/to/Data/nfl-big-data-bowl-2026-prediction")
```

Outputs are organized under `./outputs/`:
- `outputs/models/…`: Saved model weights and scalers per fold
- `outputs/results/…`: Logs, plots, metadata (`metadata.pkl`)

## Usage

### Train models
Runs the full pipeline: load data, build features, cross‑validate, and save artifacts with timestamped folders.

```zsh
# If using Poetry
poetry run python scripts/train.py

# Or with a venv
python scripts/train.py
```

The training script supports multiple architectures:
- `STTransformer` (default in `train.py` via `train_sttransformer`)
- `STTransformerDecoder` (encoder‑decoder) via `train_sttransformer_decoder`

You can switch which function runs by editing `scripts/train.py` (both functions are implemented and wired to the same data/feature pipeline).

### View results
Summaries and charts are saved under each timestamped run in `outputs/results`. Use the viewer to explore them:

```zsh
poetry run python scripts/view_results.py
# or
python scripts/view_results.py
```

## Configuration
Edit `configs/Config.py` to control data paths, devices, and hyperparameters.

- Data and outputs
  - `DATA_DIR`: absolute path to the Kaggle dataset
  - `OUTPUT_DIR`, `MODEL_DIR`, `RESULTS_DIR`: where artifacts are written
- Training
  - `SEED`, `N_FOLDS`, `BATCH_SIZE`, `EPOCHS`, `PATIENCE`, `LEARNING_RATE`
  - `WINDOW_SIZE`, `HIDDEN_DIM`, `MAX_FUTURE_HORIZON`
- Model specifics
  - STTransformer: `STTRANSFORMER_N_HEADS`, `STTRANSFORMER_N_LAYERS`, `STTRANSFORMER_USE_AE`, etc.
  - STTransformerDecoder: `STTRANSFORMER_DECODER_*` for heads/layers/teacher forcing/scheduled sampling
  - Linear/Tree/ARIMA settings are available in the same config for classical baselines
- Device
  - Automatically selects CUDA if available; falls back to CPU

## Project Structure

```
configs/
  Config.py             # Global configuration
scripts/
  train.py              # Main training script
  view_results.py       # Result visualization
src/nfl_bdb_2026/
  io.py                 # Data loading utilities
  features.py           # Feature engineering & sequence prep
  training.py           # Training loops
  cross_validation.py   # Cross‑validation orchestrator
  model.py              # STTransformer core model
  models/               # Registry & additional architectures
  loss.py, util.py, paths.py
Data/                   # Place competition data here
outputs/
  models/, results/     # Created on first run
```

## Tips & Troubleshooting

- OpenBLAS/MKL threads are limited in `scripts/train.py` to avoid oversubscription.
- If you see CUDA errors, set `GPU_ID` or force CPU by ensuring `torch.cuda.is_available()` is false.
- For quick debugging, set `DEBUG = True` in `Config.py` to reduce workload.
- Ensure `DATA_DIR` is correct and accessible; most errors stem from misconfigured paths.

## License

This project is released under the terms in `LICENSE`.

## Acknowledgments

Thanks to the NFL Big Data Bowl organizers and the Kaggle community for the dataset and challenge.
