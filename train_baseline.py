"""
Train baseline models for comparison
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import cornac
from cornac.models import MostPop, UserKNN, ItemKNN
from cornac.eval_methods import RatioSplit
from cornac.metrics import Recall, NDCG, Precision
from src.utils.data_loader import load_movielens_100k
from src.utils.evaluation import save_results
from src.config import RANDOM_SEED, METRICS_DIR
import time


def train_baseline(model_name='MostPop'):
    """
    Train a baseline recommendation model using Cornac's Experiment framework
    
    Args:
        model_name: 'MostPop', 'UserKNN', or 'ItemKNN'
    """
    print(f"\n{'='*60}")
    print(f"Training Baseline Model: {model_name}")
    print(f"{'='*60}\n")
    
    # Load data
    data = load_movielens_100k()
    
    # Create evaluation method (train/test split)
    ratio_split = RatioSplit(
        data=data,
        test_size=0.2,
        rating_threshold=0.0,
        seed=RANDOM_SEED,
        exclude_unknowns=True,
        verbose=True
    )
    
    # Select model
    if model_name == 'MostPop':
        model = MostPop()
    elif model_name == 'UserKNN':
        model = UserKNN(k=20, similarity='cosine')
    elif model_name == 'ItemKNN':
        model = ItemKNN(k=20, similarity='cosine')
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Define metrics
    metrics = [
        Recall(k=5),
        Recall(k=10),
        Recall(k=20),
        NDCG(k=5),
        NDCG(k=10),
        NDCG(k=20),
        Precision(k=5),
        Precision(k=10),
        Precision(k=20),
    ]
    
    # Run experiment
    print(f"\nTraining and evaluating {model_name}...")
    start_time = time.time()
    
    exp = cornac.Experiment(
        eval_method=ratio_split,
        models=[model],
        metrics=metrics,
        user_based=True
    )
    
    exp.run()
    training_time = time.time() - start_time
    
    print(f"\nCompleted in {training_time:.2f} seconds")
    
    # Extract results from the experiment result
    result = exp.result[0]
    
    # Get the organized results from Cornac's Result object
    results = {
        'model': model_name,
        'training_time': training_time,
    }
    
    # Extract metric values from the organized results
    # Cornac stores results in result.metric_avg_results
    if hasattr(result, 'metric_avg_results'):
        for metric_name, value in result.metric_avg_results.items():
            results[metric_name] = float(value)
    
    # Print summary
    if 'Recall@10' in results:
        print(f"\nResults for {model_name}:")
        print(f"  Recall@10: {results.get('Recall@10', 'N/A'):.4f}")
        print(f"  NDCG@10: {results.get('NDCG@10', 'N/A'):.4f}")
        print(f"  Precision@10: {results.get('Precision@10', 'N/A'):.4f}")
    
    # Save results
    save_path = METRICS_DIR / f"{model_name}_results.json"
    save_results(results, str(save_path))
    
    return model, results


if __name__ == "__main__":
    # Train all baseline models
    for model_name in ['MostPop', 'UserKNN', 'ItemKNN']:
        try:
            train_baseline(model_name)
            print("\n")
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            print("\n")
