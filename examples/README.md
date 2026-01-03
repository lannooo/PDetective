# PDetective Examples

This directory contains example scripts demonstrating how to use PDetective.

## Available Examples

### 1. `detect.py` - Inference Script

Detect partial deepfakes in audio files using a trained model.

```bash
python examples/detect.py \
    --audio path/to/audio.wav \
    --model models/baseline_model.pth \
    --visualize
```

**Arguments:**
- `--audio`: Path to input audio file (required)
- `--model`: Path to trained model checkpoint
- `--config`: Path to configuration file
- `--output`: Path to save detection results
- `--visualize`: Enable visualization of results

### 2. `train.py` - Training Script

Train a partial deepfake detection model.

```bash
python examples/train.py \
    --config configs/default_config.yaml \
    --data data/raw \
    --output models
```

**Arguments:**
- `--config`: Path to configuration file
- `--data`: Path to dataset
- `--output`: Path to save trained models
- `--resume`: Path to checkpoint to resume training
- `--gpu`: GPU device ID (-1 for CPU)

## Coming Soon

More examples will be added:
- `evaluate.py`: Evaluate model performance on test set
- `visualize.py`: Visualize model predictions and features
- `export.py`: Export model for deployment
- `demo.py`: Interactive demo with real-time detection

## Notes

⚠️ These are placeholder scripts showing the intended API design.  
The actual implementation is coming soon!

---

For more information, see the [documentation](../docs/).
