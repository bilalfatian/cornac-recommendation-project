"""
Train your main recommendation model
This is a template - you'll implement your chosen model here
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import cornac
from cornac.models import BPR  # Example model
from cornac.eval_methods import RatioSplit
from cornac.metrics import Recall, NDCG, Precision
from src.utils.data_loader import load_movielens_100k
from src.utils.evaluation import save_results
from src.config import RANDOM_SEED, EMBEDDING_DIM, LEARNING_RATE, N_EPOCHS, METRICS_DIR
import time


def train_main_model():
    """
    Train your main model (GCN, LightGCN, AutoEncoder, etc.)
    TODO: Replace with your chosen model implementation
    """
    print(f"\n{'='*60}")
    print(f"Training Main Model (BPR)")
    print(f"{'='*60}\n")
    
    # Load data
    data = load_movielens_100k()
    
    # Create evaluation method
    ratio_split = RatioSplit(
        data=data,
        test_size=0.2,
        rating_threshold=0.0,
        seed=RANDOM_SEED,
        exclude_unknowns=True,
        verbose=True
    )
    
    # TODO: Replace BPR with your chosen model
    model = BPR(
        k=EMBEDDING_DIM,
        max_iter=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        lambda_reg=0.01,
        seed=RANDOM_SEED,
        verbose=True
    )
    
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
    print(f"\nTraining and evaluating model...")
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
    
    # Get the organized results
    results = {
        'model': 'BPR',
        'training_time': training_time,
    }
    
    # Extract metric values from the organized results
    if hasattr(result, 'metric_avg_results'):
        for metric_name, value in result.metric_avg_results.items():
            results[metric_name] = float(value)
    
    # Print summary
    if 'Recall@10' in results:
        print(f"\nResults:")
        print(f"  Recall@10: {results.get('Recall@10', 'N/A'):.4f}")
        print(f"  NDCG@10: {results.get('NDCG@10', 'N/A'):.4f}")
        print(f"  Precision@10: {results.get('Precision@10', 'N/A'):.4f}")
    
    # Save results
    save_path = METRICS_DIR / "BPR_results.json"
    save_results(results, str(save_path))
    
    return model, results


if __name__ == "__main__":
    train_main_model()
