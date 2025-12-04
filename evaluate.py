"""
Centralized evaluation script for all models
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
from src.config import METRICS_DIR
import matplotlib.pyplot as plt
import seaborn as sns


def compare_models():
    """
    Load and compare all model results
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60 + "\n")
    
    results_files = list(METRICS_DIR.glob("*_results.json"))
    
    if not results_files:
        print("No results found. Run training scripts first.")
        return
    
    all_results = {}
    for file in results_files:
        with open(file, 'r') as f:
            data = json.load(f)
            model_name = data.get('model', file.stem)
            all_results[model_name] = data
    
    # Print comparison table
    print(f"{'Model':<20} {'Recall@10':<12} {'NDCG@10':<12} {'Time (s)':<12}")
    print("-" * 60)
    
    for model_name, results in all_results.items():
        recall = results.get('Recall@10', 'N/A')
        ndcg = results.get('NDCG@10', 'N/A')
        time_taken = results.get('training_time', 'N/A')
        
        recall_str = f"{recall:.4f}" if isinstance(recall, float) else recall
        ndcg_str = f"{ndcg:.4f}" if isinstance(ndcg, float) else ndcg
        time_str = f"{time_taken:.2f}" if isinstance(time_taken, float) else time_taken
        
        print(f"{model_name:<20} {recall_str:<12} {ndcg_str:<12} {time_str:<12}")
    
    return all_results


if __name__ == "__main__":
    compare_models()
