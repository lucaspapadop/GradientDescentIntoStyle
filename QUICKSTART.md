# Quick Start Guide

## Running on Google Colab (Recommended)

### Step 1: Setup (5 minutes)
1. Open the notebook in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/deepfashion2-adaptive-clustering/blob/main/DeepFashion2AdaptiveClustering.ipynb)
2. Go to Runtime → Change runtime type → Select **T4 GPU**
3. Run the first cells to check GPU availability

### Step 2: Get Kaggle API Key (2 minutes)
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. Download `kaggle.json` file

### Step 3: Run Training
1. Execute cells in order
2. When prompted, upload your `kaggle.json` file
3. First run downloads dataset (~20-40 minutes, one-time only)
4. Subsequent runs use cached data from Google Drive

### Step 4: Monitor Progress
Training will show:
- Feature extraction progress bars
- Clustering results (silhouette score, novel samples)
- Epoch summaries (loss, accuracy, cluster count)
- Checkpoints saved every 2 epochs

### Expected Timeline
- **First run**: ~2-3 hours (includes dataset download)
- **Subsequent runs**: ~1-1.5 hours (uses cached features)

---

## Running Locally (Advanced)

### Prerequisites
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+ installed
- 20GB+ free disk space

### Installation
```bash
git clone https://github.com/yourusername/deepfashion2-adaptive-clustering.git
cd deepfashion2-adaptive-clustering
pip install -r requirements.txt
```

### Download Dataset
```bash
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d thusharanair/deepfashion2-original-with-dataframes
unzip deepfashion2-original-with-dataframes.zip -d datasets/
```

### Run Training
```bash
jupyter notebook DeepFashion2AdaptiveClustering.ipynb
# Or convert to .py and run as script
```

---

## Common Issues

### "No GPU detected"
- In Colab: Runtime → Change runtime type → GPU (T4)
- Locally: Check `nvidia-smi` and CUDA installation

### "Dataset not found"
- Ensure kaggle.json is uploaded/configured
- Check Google Drive mount: `/content/drive/MyDrive/DeepFashion_Project/`

### "Out of memory"
- Reduce `BATCH_SIZE` in config (try 16 or 8)
- Close other Colab tabs

### "Slow training"
- First run: Dataset download takes 20-40 min (normal)
- Check GPU utilization with `!nvidia-smi` in Colab

---

## Modifying Hyperparameters

Edit the `Config` class in the notebook:

```python
class Config:
    # Try smaller batch size if OOM
    BATCH_SIZE = 16  # Default: 32
    
    # Train longer for better convergence
    NUM_EPOCHS = 12  # Default: 8
    
    # Start with fewer clusters
    INITIAL_K = 8  # Default: 10
    
    # Allow more clusters
    MAX_K = 30  # Default: 25
```

---

## Next Steps

After training completes:
1. Check `checkpoints/` folder for saved models
2. Review `training_history.json` for metrics
3. Visualize cluster distributions (see evaluation cells)
4. Export embeddings for further analysis

For detailed documentation, see [README.md](README.md)
