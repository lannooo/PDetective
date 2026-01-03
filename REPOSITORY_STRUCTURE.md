# PDetective Repository Structure

This document provides an overview of the repository structure and all files created for the PDetective project.

## Repository Overview

PDetective is an initial research repository for studying partial deepfake (partially spoofed) speech detection. The repository is organized to support machine learning research, with clear separation between source code, data, models, experiments, and documentation.

## Files and Directories

### Root Level Files

| File | Purpose |
|------|---------|
| `README.md` | Main project documentation and overview |
| `LICENSE` | MIT License for the project |
| `CONTRIBUTING.md` | Guidelines for contributing to the project |
| `CHANGELOG.md` | Record of changes and version history |
| `QUICKSTART.md` | Quick start guide for developers |
| `requirements.txt` | Python package dependencies |
| `setup.py` | Package installation configuration |
| `.gitignore` | Git ignore rules for Python projects |

### Source Code (`src/`)

```
src/
├── __init__.py              # Package initialization (v0.1.0)
├── models/
│   └── __init__.py         # Model architectures (placeholder)
├── datasets/
│   └── __init__.py         # Dataset loaders (placeholder)
├── training/
│   └── __init__.py         # Training utilities (placeholder)
├── evaluation/
│   └── __init__.py         # Evaluation metrics (placeholder)
└── utils/
    └── __init__.py         # Utility functions (placeholder)
```

**Purpose**: Core implementation code for the framework. All modules currently contain placeholders with "code coming soon" notes.

### Examples (`examples/`)

```
examples/
├── README.md               # Examples documentation
├── train.py               # Training script example
└── detect.py              # Inference script example
```

**Purpose**: Demonstration scripts showing intended API usage. Scripts are functional placeholders that show help messages and expected workflow.

### Configuration (`configs/`)

```
configs/
└── default_config.yaml    # Example configuration file
```

**Purpose**: YAML configuration files for experiments. Includes settings for model architecture, training, evaluation, and experiment tracking.

### Documentation (`docs/`)

```
docs/
├── README.md              # Documentation index
├── getting_started.md     # Installation and setup guide
└── architecture.md        # System architecture overview
```

**Purpose**: Detailed project documentation including guides, architecture descriptions, and API references.

### Notebooks (`notebooks/`)

```
notebooks/
└── getting_started.ipynb  # Interactive tutorial notebook
```

**Purpose**: Jupyter notebooks for interactive exploration, analysis, and tutorials. The getting started notebook includes placeholder cells demonstrating the intended workflow.

### Tests (`tests/`)

```
tests/
└── test_basic.py          # Basic unit tests
```

**Purpose**: Unit and integration tests. Currently includes basic tests for package import and infrastructure.

### Data Directories

```
data/
├── raw/                   # Raw audio files (not tracked)
├── processed/             # Preprocessed features (not tracked)
└── splits/                # Train/val/test splits (not tracked)
```

**Purpose**: Data storage. These directories are excluded from git via `.gitignore` but include `.gitkeep` files to maintain structure.

### Model Storage

```
models/                    # Saved model checkpoints (not tracked)
```

**Purpose**: Storage for trained model files. Excluded from git but structure is preserved.

### Experiments

```
experiments/               # Experiment logs and results (not tracked)
```

**Purpose**: Storage for experiment outputs, logs, and results.

## Key Features

### ✅ Implemented

- Complete project structure with clear organization
- Comprehensive documentation and guides
- Example scripts with clear API demonstrations
- Configuration templates
- Development infrastructure (tests, linting setup)
- Proper Python package structure with setup.py
- Contributing guidelines and code of conduct
- MIT License
- `.gitignore` configured for Python/ML projects

### 🚧 Coming Soon (as noted in files)

- Model implementations (CNN, RNN, Transformer architectures)
- Dataset loaders for ASVspoof and other datasets
- Training pipeline and utilities
- Evaluation metrics (EER, AUC, etc.)
- Feature extraction utilities
- Pre-trained models
- Comprehensive unit tests
- Tutorial notebooks
- Full documentation

## Getting Started

For developers starting work on this repository:

1. Read `QUICKSTART.md` for development setup
2. Review `CONTRIBUTING.md` for contribution guidelines
3. Check `docs/` for detailed documentation
4. Examine `examples/` for intended API usage
5. Review `configs/default_config.yaml` for configuration structure

## Notes

- All placeholder files clearly indicate "code coming soon"
- Structure follows Python best practices
- Modular design allows independent development of components
- Configuration-driven approach for reproducibility
- Ready for immediate development work

---

**Last Updated**: January 3, 2024  
**Version**: 0.1.0  
**Status**: Initial structure complete, implementation in progress
