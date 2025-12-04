"""
Configuration file for the project
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Experiments directory
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Random seed for reproducibility
RANDOM_SEED = 42

# Train/Val/Test split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Model parameters (you'll adjust these later)
EMBEDDING_DIM = 64
LEARNING_RATE = 0.001
BATCH_SIZE = 128
N_EPOCHS = 50

# Evaluation metrics
TOP_K = [5, 10, 20]
