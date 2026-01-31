"""
Core Experiment Script for Isolation Forest Paper Replication
Matches methodology from: "Revisiting randomized choices in isolation forests" (Cortes, 2021)

Methodology (paper-style protocol):
1. Datasets: As per Table 5 in paper.
2. Training: Unsupervised on full dataset (no split).
3. Evaluation: Score full dataset; ROC-AUC and PR-AUC on full labels.
4. Repetitions: 10 random seeds by default (0..9).
5. Scaling: Applied for distance-based methods (LOF, OCSVM).
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import time
import warnings
import sys
import os

# Filter warnings
warnings.filterwarnings('ignore')

# Defaults / config
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_DATASETS = [
    "arrhythmia",
    "pima",
    "spambase",
    "satellite",
    "pendigits",
    "annthyroid",
    "mnist",
]
DEFAULT_SEED_COUNT = range(10)
DEFAULT_OUTPUT_DIR = "."

# Model groups
ISOTREE_MODELS = {"IF", "IF-u", "EIF", "SCiF", "SCiF-u", "FCF"}

# Import anomaly detection models
try:
    from isotree import IsolationForest
except ImportError:
    print("Error: isotree not installed. Please install it.")
    sys.exit(1)

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from scipy.io import loadmat

def load_odds_mat(path: str):
    mat = loadmat(path)

    # ODDS mats typically store: X (n×d), y (n×1) or (n,)
    X = mat.get("X", None)
    y = mat.get("y", None)

    if X is None or y is None:
        raise ValueError(f"Missing X or y in {path}. Keys: {list(mat.keys())}")

    # y often comes as shape (n,1) float; convert to 1D int {0,1}
    y = np.asarray(y).reshape(-1)
    # Some ODDS y labels are {1, -1} or {0,1}; normalize to {0,1}
    if set(np.unique(y)).issuperset({-1, 1}):
        y = (y == -1).astype(int)
    else:
        y = (y != 0).astype(int)

    X = np.asarray(X, dtype=np.float32)
    return X, y

def load_dataset(dataset_name):
    """
    Load datasets to match Table 5 in the paper.
    """
    print(f"Loading {dataset_name}...")

    if dataset_name == "arrhythmia":
        # load .mat file from folder
        X, y_binary = load_odds_mat(os.path.join(BASE_DIR, "datasets", "arrhythmia.mat"))
            
    elif dataset_name == "pima":
        # Table 5: 768 rows, 8 cols, 35% outliers
        mat_path = os.path.join(BASE_DIR, "datasets", "pima.mat")
        if os.path.exists(mat_path):
            X, y_binary = load_odds_mat(mat_path)
        else:
            data = fetch_openml('diabetes', version=1, parser='auto')
            X = np.array(data.data, dtype=np.float32)
            y = np.array(data.target)
            y_binary = (y == 'tested_positive').astype(int)
        
    elif dataset_name == "spambase":
        # Table 5: 4601 rows, 57 cols, 39.4% outliers
        mat_path = os.path.join(BASE_DIR, "datasets", "spambase.mat")
        if os.path.exists(mat_path):
            X, y_binary = load_odds_mat(mat_path)
        else:
            data = fetch_openml('spambase', version=1, parser='auto')
            X = np.array(data.data, dtype=np.float32)
            y = np.array(data.target)
            y_binary = (y == '1').astype(int)
        
    elif dataset_name == "satellite":
        X, y_binary = load_odds_mat(os.path.join(BASE_DIR, "datasets", "satellite.mat"))
            
    elif dataset_name == "mnist":
        X, y_binary = load_odds_mat(os.path.join(BASE_DIR, "datasets", "mnist.mat"))
            
    elif dataset_name == "pendigits":
        X, y_binary = load_odds_mat(os.path.join(BASE_DIR, "datasets", "pendigits.mat"))

    elif dataset_name == "annthyroid":
        X, y_binary = load_odds_mat(os.path.join(BASE_DIR, "datasets", "annthyroid.mat"))

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Impute missing
    if np.any(np.isnan(X)):
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    
    print(f"  Shape: {X.shape}")
    print(f"  Outliers: {np.sum(y_binary)} ({100*np.mean(y_binary):.2f}%)")
    
    return X, y_binary

def get_model_factories():
    """
    Return dictionary of factories. Each factory takes a 'seed' argument
    and returns an instantiated model.
    """
    sample_size = 256
    standard_depth = 8
    factories = {
        'IF': lambda s: IsolationForest(
            ntrees=100, sample_size=sample_size, ndim=1, max_depth=standard_depth,
            prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0,
            nthreads=-1, random_seed=s
        ),
        # IF-u: same as IF but with 200 trees and unlimited depth
        'IF-u': lambda s: IsolationForest(
            ntrees=200, sample_size=sample_size, ndim=1, max_depth=None,
            prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0,
            nthreads=-1, random_seed=s
        ),
        # Extended Isolation Forest variants (ndim=2)
        "EIF": lambda s: IsolationForest(
            ntrees=100, sample_size=sample_size, ndim=2, max_depth=standard_depth,
            # Keep it RANDOM. Do not turn on gain-based selection.
            prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0,
            nthreads=-1, random_seed=s,
        ),
        # SCiForest: averaged gain, 10 trials
        'SCiF': lambda s: IsolationForest(
            ntrees=100, sample_size=sample_size, ndim=2, ntry=10, max_depth=standard_depth,
            prob_pick_avg_gain=1.0, prob_pick_pooled_gain=0.0,
            penalize_range=True,
            nthreads=-1, random_seed=s
        ),
        # SCiF-u: 200 trees, unlimited depth (paper variant)
        'SCiF-u': lambda s: IsolationForest(
            ntrees=200, sample_size=sample_size, ndim=2, ntry=10, max_depth=None,
            prob_pick_avg_gain=1.0, prob_pick_pooled_gain=0.0,
            penalize_range=True,
            nthreads=-1, random_seed=s
        ),
        # Fair-Cut Forest (pooled gain, ntry=1, 200 trees, unlimited depth)
        'FCF': lambda s: IsolationForest(
            ntrees=200, sample_size=sample_size, ndim=2, ntry=1, max_depth=None,
            prob_pick_pooled_gain=1.0, prob_pick_avg_gain=0.0,
            nthreads=-1, random_seed=s
        ),
        'LOF': lambda s: LocalOutlierFactor(
            n_neighbors=20, contamination='auto', novelty=True, n_jobs=-1
        ),
        'OCSVM-rbf': lambda s: OneClassSVM(kernel='rbf', gamma='auto', nu=0.1),
        'OCSVM-linear': lambda s: OneClassSVM(kernel='linear', nu=0.1),
    }
    return factories

def needs_scaling(model_name):
    return model_name.startswith('LOF') or model_name.startswith('OCSVM')

def score_model(model_name, model, X):
    """
    Return anomaly scores with higher = more anomalous.
    """
    if model_name in ISOTREE_MODELS:
        try:
            scores = model.predict(X, output="score")
        except (TypeError, ValueError):
            try:
                scores = model.decision_function(X)
            except (AttributeError, TypeError):
                scores = model.predict(X)
        scores = np.asarray(scores).reshape(-1)
        return scores
    if model_name.startswith('LOF'):
        return -model.decision_function(X)
    if model_name.startswith('OCSVM'):
        return -model.decision_function(X)
    scores = model.predict(X)
    return np.asarray(scores).reshape(-1)

def test_model(model_name, model_factory, X, y, seeds):
    """
    Test a single model using the paper-style protocol:
    fit and score on full X for each seed.
    """
    roc_scores = []
    pr_scores = []
    times = []
    
    print(f"  {model_name:12s}", end=' ')
    
    for seed in seeds:
        try:
            # 1. Feature Scaling (Important for LOF/OCSVM)
            if needs_scaling(model_name):
                scaler = StandardScaler()
                X_eval = scaler.fit_transform(X)
            else:
                X_eval = X

            # 2. Instantiate and Fit on full data
            start_time = time.time()
            model = model_factory(seed)
            model.fit(X_eval)
            
            # 3. Predict scores (Higher = More Anomalous)
            scores = score_model(model_name, model, X_eval)
            if np.unique(scores).size <= 10:
                print(f"Warning: {model_name} returned few unique scores ({np.unique(scores).size}).", end=' ')
                
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # 4. Score on full dataset
            roc = roc_auc_score(y, scores)
            pr = average_precision_score(y, scores)
            
            roc_scores.append(roc)
            pr_scores.append(pr)
            
        except Exception as e:
            print(f"[Err seed={seed}: {e}]", end=' ')
            # If one seed fails, likely all will. Return None to skip.
            return None

    if not roc_scores:
        print("FAILED")
        return None
        
    mean_roc = np.mean(roc_scores)
    std_roc = np.std(roc_scores)
    mean_pr = np.mean(pr_scores)
    std_pr = np.std(pr_scores)
    mean_time = np.mean(times)
    
    print(f"ROC: {mean_roc:.4f}±{std_roc:.4f}, PR: {mean_pr:.4f}±{std_pr:.4f}, Time: {mean_time:.4f}s")
    
    return {
        'model': model_name,
        'roc_auc_mean': mean_roc,
        'roc_auc_std': std_roc,
        'pr_auc_mean': mean_pr,
        'pr_auc_std': std_pr,
        'train_time': mean_time
    }

def run_sanity_check(dataset_name, model_name, model_factory, X, y, seed):
    print("\n" + "-" * 80)
    print(f"SANITY CHECK: {dataset_name} | {model_name} | seed={seed}")
    print("-" * 80)
    if needs_scaling(model_name):
        scaler = StandardScaler()
        X_eval = scaler.fit_transform(X)
    else:
        X_eval = X
    model = model_factory(seed)
    model.fit(X_eval)
    scores = score_model(model_name, model, X_eval)
    roc = roc_auc_score(y, scores)
    pr = average_precision_score(y, scores)
    roc_inv = roc_auc_score(y, -scores)
    pr_inv = average_precision_score(y, -scores)
    roc_pref = "scores" if roc >= roc_inv else "-scores"
    pr_pref = "scores" if pr >= pr_inv else "-scores"
    print(f"ROC-AUC: scores={roc:.4f} vs -scores={roc_inv:.4f} -> {roc_pref}")
    print(f"PR-AUC:  scores={pr:.4f} vs -scores={pr_inv:.4f} -> {pr_pref}")

def compare_models_on_dataset(dataset_name, seeds, model_subset=None):
    """Run all models on a dataset"""
    print(f"\n" + "="*80)
    print(f"DATASET: {dataset_name.upper()}")
    print("="*80)
    
    X, y = load_dataset(dataset_name)
    if X is None:
        return []
    
    factories = get_model_factories()
    if model_subset:
        factories = {k: v for k, v in factories.items() if k in model_subset}
    
    results = []
    for name, factory in factories.items():
        res = test_model(name, factory, X, y, seeds)
        if res:
            res['dataset'] = dataset_name
            results.append(res)
            
    return results

if __name__ == "__main__":
    datasets = DEFAULT_DATASETS
    seeds = DEFAULT_SEED_COUNT
    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    test_models = [
        'IF', 'IF-u', 'EIF',
        'SCiF', 'SCiF-u', 'FCF',
        'LOF', 'OCSVM-rbf', 'OCSVM-linear'
    ]

    factories = get_model_factories()

    # Setup CSV
    output_csv = os.path.join(output_dir, "anomaly_detection_results.csv")
    if os.path.exists(output_csv):
        os.remove(output_csv)

    pd.DataFrame(columns=['dataset', 'model',
                            'roc_auc_mean', 'roc_auc_std',
                            'pr_auc_mean', 'pr_auc_std',
                            'train_time']).to_csv(output_csv, index=False)

    for dataset in datasets:
        try:
            current_models = test_models.copy()
            dataset_results = compare_models_on_dataset(dataset, seeds, model_subset=current_models)

            if dataset_results:
                df = pd.DataFrame(dataset_results)
                # Enforce column order to match CSV header
                df = df[['dataset', 'model', 'roc_auc_mean', 'roc_auc_std', 'pr_auc_mean', 'pr_auc_std', 'train_time']]
                df.to_csv(output_csv, mode='a', header=False, index=False)
                print(f"Saved results for {dataset}")

        except Exception as e:
            print(f"\nERROR with dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()



