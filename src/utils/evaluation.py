"""
Evaluation utilities for recommendation models
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from typing import Dict, List
import cornac
from cornac.metrics import Recall, NDCG, Precision


def evaluate_model(model, train_set, test_set, metrics=['recall', 'ndcg'], k_values=[5, 10, 20]):
    """
    Evaluate a Cornac model using Cornac's Experiment framework
    
    Args:
        model: Trained Cornac model
        train_set: Training dataset (needed for evaluation context)
        test_set: Test dataset
        metrics: List of metrics to compute
        k_values: List of k values for top-k metrics
        
    Returns:
        results: Dictionary of metric scores
    """
    metric_objects = []
    
    for k in k_values:
        if 'recall' in metrics:
            metric_objects.append(Recall(k=k))
        if 'ndcg' in metrics:
            metric_objects.append(NDCG(k=k))
        if 'precision' in metrics:
            metric_objects.append(Precision(k=k))
    
    # Use Cornac's Experiment for proper evaluation
    exp = cornac.Experiment(
        eval_method=cornac.eval_methods.RatioSplit(
            data=train_set.uir_tuple,
            test_size=0.2,
            rating_threshold=0.0,
            seed=42,
            exclude_unknowns=True,
            verbose=False
        ),
        models=[model],
        metrics=metric_objects,
        user_based=True
    )
    
    # Get the result
    result = exp.result[0]
    
    # Extract metrics
    results = {}
    for metric in metric_objects:
        metric_name = f"{metric.name}@{metric.k}"
        score = result.metric_user_results[metric.name]
        results[metric_name] = float(score)
        print(f"{metric_name}: {score:.4f}")
    
    return results


def evaluate_model_simple(model, test_set, k_values=[5, 10, 20]):
    """
    Simplified evaluation that works with trained models
    
    Args:
        model: Trained Cornac model
        test_set: Test dataset (Cornac Dataset object)
        k_values: List of k values for top-k metrics
        
    Returns:
        results: Dictionary of metric scores
    """
    from cornac.metrics import Recall, NDCG, Precision
    
    results = {}
    
    # Get test users
    test_users = set()
    for uid, iid, rating in test_set.uir_tuple:
        test_users.add(uid)
    
    # Compute metrics for each k
    for k in k_values:
        recall_metric = Recall(k=k)
        ndcg_metric = NDCG(k=k)
        precision_metric = Precision(k=k)
        
        recall_scores = []
        ndcg_scores = []
        precision_scores = []
        
        for user_idx in test_users:
            try:
                # Get ground truth items for this user
                gt_items = test_set.user_data[user_idx]
                
                # Get recommendations
                recommendations = model.rank(user_idx)
                top_k_items = recommendations[:k]
                
                # Compute metrics
                # Recall
                hits = len(set(top_k_items) & set(gt_items))
                recall = hits / min(len(gt_items), k) if len(gt_items) > 0 else 0
                recall_scores.append(recall)
                
                # Precision
                precision = hits / k if k > 0 else 0
                precision_scores.append(precision)
                
                # NDCG (simplified)
                dcg = sum([1.0 / np.log2(i + 2) for i, item in enumerate(top_k_items) if item in gt_items])
                idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(gt_items), k))])
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_scores.append(ndcg)
                
            except:
                continue
        
        # Average across users
        if recall_scores:
            results[f'Recall@{k}'] = np.mean(recall_scores)
            print(f"Recall@{k}: {results[f'Recall@{k}']:.4f}")
        
        if ndcg_scores:
            results[f'NDCG@{k}'] = np.mean(ndcg_scores)
            print(f"NDCG@{k}: {results[f'NDCG@{k}']:.4f}")
        
        if precision_scores:
            results[f'Precision@{k}'] = np.mean(precision_scores)
            print(f"Precision@{k}: {results[f'Precision@{k}']:.4f}")
    
    return results


def save_results(results: Dict, filepath: str):
    """Save evaluation results to file"""
    import json
    from pathlib import Path
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {filepath}")


if __name__ == "__main__":
    print("Evaluation utilities loaded successfully!")
