# Quick Start Guide for Developers

Welcome to PDetective! This guide will help you get started with development.

## Repository Overview

This repository is set up for research on partial deepfake speech detection. The structure is designed to be:
- **Modular**: Easy to add new models, datasets, and evaluation metrics
- **Research-friendly**: Jupyter notebooks for experimentation
- **Production-ready**: Proper package structure for deployment

## What's Already Set Up

✅ Project directory structure  
✅ Python package scaffolding  
✅ Requirements and dependencies  
✅ Documentation framework  
✅ Example configuration files  
✅ Test infrastructure  
✅ Git configuration (.gitignore)  
✅ Contributing guidelines  
✅ License (MIT)  

## Next Steps for Development

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 2. Directory Guide

- **`src/`**: Put your implementation code here
  - `models/`: Neural network architectures
  - `datasets/`: Data loading and preprocessing
  - `training/`: Training loops and utilities
  - `evaluation/`: Metrics and evaluation scripts
  - `utils/`: Helper functions

- **`configs/`**: Configuration files for experiments
  - Use YAML for configs
  - See `default_config.yaml` for template

- **`notebooks/`**: Jupyter notebooks for analysis
  - Exploratory data analysis
  - Model visualization
  - Results analysis

- **`tests/`**: Unit and integration tests
  - Follow pytest conventions
  - Test each module as you build

- **`docs/`**: Documentation
  - Keep docs up to date with code
  - Add examples and tutorials

### 3. Development Workflow

1. **Create a new feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Implement your changes**
   - Write code in appropriate module
   - Add tests for new functionality
   - Update documentation

3. **Format and lint your code**
   ```bash
   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   ```

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```

5. **Commit and push**
   ```bash
   git add .
   git commit -m "Descriptive commit message"
   git push origin feature/your-feature-name
   ```

### 4. Key Areas to Implement

**Priority 1: Core Functionality**
- [ ] Audio preprocessing pipeline
- [ ] Feature extraction (MFCC, spectrogram)
- [ ] Basic CNN/RNN model
- [ ] Training loop
- [ ] Evaluation metrics (EER, AUC)

**Priority 2: Data Handling**
- [ ] Dataset loaders for ASVspoof
- [ ] Data augmentation
- [ ] Train/val/test splitting
- [ ] Batch processing

**Priority 3: Advanced Features**
- [ ] Temporal localization
- [ ] Advanced model architectures
- [ ] Visualization tools
- [ ] Pre-trained model support

**Priority 4: Polish**
- [ ] Comprehensive documentation
- [ ] Tutorial notebooks
- [ ] Example scripts
- [ ] Benchmark results

### 5. Coding Standards

- **Style**: Follow PEP 8
- **Docstrings**: Use Google or NumPy style
- **Type hints**: Use where appropriate
- **Comments**: Explain "why", not "what"
- **Tests**: Aim for >80% coverage

### 6. Experiment Tracking

Consider using:
- TensorBoard for visualization
- Weights & Biases for experiment tracking (optional)
- Config files for reproducibility

### 7. Data Management

- **Never commit large files** (audio, models)
- Use `.gitignore` appropriately
- Keep `data/` directory local
- Document dataset sources in docs

### 8. Getting Help

- Check existing [Issues](https://github.com/lannooo/PDetective/issues)
- Read [Contributing Guide](CONTRIBUTING.md)
- Open a new issue for questions

## Tips for Success

1. **Start simple**: Implement basic functionality first
2. **Test early**: Write tests as you code
3. **Document continuously**: Don't leave docs for later
4. **Use configs**: Make experiments reproducible
5. **Version models**: Save checkpoints regularly
6. **Iterate**: Build → Test → Improve

## Common Commands

```bash
# Install in dev mode
pip install -e .

# Run tests
pytest tests/ -v

# Format code
black .

# Sort imports
isort .

# Lint
flake8 .

# Run notebook
jupyter notebook notebooks/getting_started.ipynb

# Train model (once implemented)
python -m src.training.train --config configs/default_config.yaml
```

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [librosa Documentation](https://librosa.org/doc/latest/)
- [ASVspoof Challenge](https://www.asvspoof.org/)

---

Happy coding! 🚀
