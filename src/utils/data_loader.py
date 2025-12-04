"""
Data loading utilities for Cornac datasets
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import cornac
from cornac.datasets import movielens
import numpy as np
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED


def load_movielens_100k():
    """
    Load MovieLens 100K dataset
    
    Returns:
        data: List of (user_id, item_id, rating) tuples
    """
    print("Loading MovieLens 100K dataset...")
    data = movielens.load_feedback(variant="100K")
    print(f"Loaded {len(data)} ratings")
    return data


def load_movielens_1m():
    """
    Load MovieLens 1M dataset
    
    Returns:
        data: List of (user_id, item_id, rating) tuples
    """
    print("Loading MovieLens 1M dataset...")
    data = movielens.load_feedback(variant="1M")
    print(f"Loaded {len(data)} ratings")
    return data


def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=RANDOM_SEED):
    """
    Split data into train/val/test sets
    
    Args:
        data: List of (user_id, item_id, rating) tuples
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
        
    Returns:
        train_data, val_data, test_data
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    np.random.seed(seed)
    
    # Shuffle data
    data_array = np.array(data)
    np.random.shuffle(data_array)
    
    # Calculate split indices
    n = len(data_array)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    # Split
    train_data = data_array[:train_end].tolist()
    val_data = data_array[train_end:val_end].tolist()
    test_data = data_array[val_end:].tolist()
    
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Test the data loader
    data = load_movielens_100k()
    train, val, test = split_data(data)
    print("Data loading successful!")
