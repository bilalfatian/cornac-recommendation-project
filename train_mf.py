"""
Train a Matrix Factorization (MF) baseline and save metrics for evaluate.py
Outputs: results/metrics/MF_results.json (or METRICS_DIR/MF_results.json)
"""

import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import cornac
from cornac.models import MF
from cornac.eval_methods import RatioSplit
from cornac.metrics import Recall, NDCG, Precision

from src.utils.data_loader import load_movielens_100k
from src.utils.evaluation import save_results
from src.config import RANDOM_SEED, EMBEDDING_DIM, LEARNING_RATE, N_EPOCHS, METRICS_DIR


def train_mf():
    print("\n" + "=" * 60)
    print("Training Baseline: Matrix Factorization (MF)")
    print("=" * 60 + "\n")

    # Load data (user-item interactions)
    data = load_movielens_100k()

    # Train/test split
    ratio_split = RatioSplit(
        data=data,
        test_size=0.2,
        rating_threshold=0.0,   # treat all ratings as positive interactions
        seed=RANDOM_SEED,
        exclude_unknowns=True,
        verbose=True
    )

    # MF model (standard baseline)
    # Notes:
    # - k = latent dimension
    # - max_iter = number of training epochs/iterations
    # - learning_rate and lambda_reg control SGD updates/regularization
    model = MF(
        k=EMBEDDING_DIM,
        max_iter=N_EPOCHS,
        learning_rate=LEARNING_RATE,
        lambda_reg=0.01,
        seed=RANDOM_SEED,
        verbose=True
    )

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

    print("\nTraining and evaluating MF...")
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

    # Cornac Experiment results
    result = exp.result[0]

    results = {
        "model": "MF",
        "training_time": float(training_time),
        "k": int(EMBEDDING_DIM),
        "max_iter": int(N_EPOCHS),
        "learning_rate": float(LEARNING_RATE),
        "lambda_reg": 0.01,
        "seed": int(RANDOM_SEED),
    }

    # Extract averaged metrics
    if hasattr(result, "metric_avg_results"):
        for metric_name, value in result.metric_avg_results.items():
            try:
                results[metric_name] = float(value)
            except Exception:
                results[metric_name] = value

    # Print the key ones required by the assignment
    print("\nResults (TEST):")
    if "Recall@10" in results:
        print(f"  Recall@10:    {results['Recall@10']:.4f}")
    if "NDCG@10" in results:
        print(f"  NDCG@10:      {results['NDCG@10']:.4f}")
    if "Precision@10" in results:
        print(f"  Precision@10: {results['Precision@10']:.4f}")

    # Save so evaluate.py can find it
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = METRICS_DIR / "MF_results.json"
    save_results(results, str(save_path))
    print(f"\nSaved metrics to: {save_path}")

    return model, results


if __name__ == "__main__":
    train_mf()
