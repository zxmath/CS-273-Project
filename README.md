# NFL Big Data Bowl 2026 - Prediction

Repository for the [NFL Big Data Bowl 2026 Prediction Competition](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview).

## Structure

```
.
├── configs/          # Configuration files
├── Data/            # Data directory
├── docs/            # Documentation
├── models/          # Saved models
├── notebooks/       # Jupyter notebooks
├── reports/         # Reports and visualizations
├── scripts/         # Executable scripts
├── src/             # Source code
├── submissions/     # Submission files
└── tests/           # Tests
```

## Setup

### Using Poetry (Recommended)

1. **Install Poetry** (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Install dependencies**:
```bash
poetry install
```

3. **Activate the environment**:
```bash
poetry shell
```

4. **Run scripts**:
```bash
poetry run python scripts/train.py
```

### Using pip

```bash
pip install -r requirements.txt
pip install -e .
```

## Usage

Add your implementation here.
