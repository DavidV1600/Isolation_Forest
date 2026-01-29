"""
Core Experiment Script for Isolation Forest Paper Replication
Matches methodology from: "Revisiting randomized choices in isolation forests" (Cortes, 2021)

Methodology:
1. Datasets: As per Table 5 in paper.
2. Training: Semi-supervised (Train on Normal, Test on All).
3. Splits: Stratified 70/30.
4. Repetitions: 5 random seeds.
5. Metrics: ROC-AUC, PR-AUC.
6. Scaling: Applied for distance-based methods (LOF, OCSVM).
"""

import numpy as np
import pandas as pd
import scipy
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import time
import warnings
import sys
import os

# Filter warnings
warnings.filterwarnings('ignore')

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
        data = load_odds_mat('../datasets/arrhythmia.mat')
        X = data[0]
        y = data[1]
        y_binary = y.astype(int)
            
    elif dataset_name == "pima":
        # Table 5: 768 rows, 8 cols, 35% outliers
        data = fetch_openml('diabetes', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        y_binary = (y == 'tested_positive').astype(int)
        
    elif dataset_name == "spambase":
        # Table 5: 4601 rows, 57 cols, 39.4% outliers
        data = fetch_openml('spambase', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        y_binary = (y == '1').astype(int)
        
    elif dataset_name == "satellite":
        data = load_odds_mat('../datasets/satellite.mat')
        X = data[0]
        y = data[1]
        y_binary = y.astype(int)
            
    elif dataset_name == "mnist":
        data = load_odds_mat('../datasets/mnist.mat')
        X = data[0]
        y = data[1]
        y_binary = y.astype(int)
            
    elif dataset_name == "pendigits":
        data = load_odds_mat('../datasets/pendigits.mat')
        X = data[0]
        y = data[1]
        y_binary = y.astype(int)

    elif dataset_name == "annthyroid":
        data = load_odds_mat('../datasets/annthyroid.mat')
        X = data[0]
        y = data[1]
        y_binary = y.astype(int)

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
    factories = {
        'IF': lambda s: IsolationForest(
            ntrees=100, sample_size=256, ndim=1, 
            prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0,
            nthreads=-1, random_seed=s
        ),
        'IF-U': lambda s: IsolationForest(
            ntrees=100, sample_size=256, ndim=1,
            prob_pick_pooled_gain=1.0, prob_pick_avg_gain=0.0,
            nthreads=-1, random_seed=s
        ),
        'EIF-o': lambda s: IsolationForest(
            ntrees=100, sample_size=256, ndim=2,
            prob_pick_pooled_gain=0.0, prob_pick_avg_gain=0.0,
            nthreads=-1, random_seed=s
        ),
        'EIF-t': lambda s: IsolationForest(
            ntrees=100, sample_size=256, ndim=2,
            prob_pick_pooled_gain=1.0, prob_pick_avg_gain=0.0,
            nthreads=-1, random_seed=s
        ),
        'SCiF': lambda s: IsolationForest(
            ntrees=100, sample_size=256, ndim=2, ntry=10,
            prob_pick_avg_gain=1.0, prob_pick_pooled_gain=0.0,
            nthreads=-1, random_seed=s
        ),
        'SCiF-u': lambda s: IsolationForest(
            ntrees=100, sample_size=256, ndim=2, ntry=10,
            prob_pick_avg_gain=1.0, prob_pick_pooled_gain=1.0,
            nthreads=-1, random_seed=s
        ),
        'FCF': lambda s: IsolationForest(
            ntrees=200, sample_size=256, ndim=2, ntry=1,
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

def test_model(model_name, model_factory, X, y):
    """
    Test a single model using stratified train/test split.
    Protocol: Semi-supervised (Train on Normal, Test on All/Test-Set).
    Seeds: 5 repetitions.
    """
    seeds = [42, 1, 2, 3, 4]
    roc_scores = []
    pr_scores = []
    times = []
    
    print(f"  {model_name:12s}", end=' ')
    
    for seed in seeds:
        try:
            # 1. Stratified split (70% train, 30% test)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )
            
            # 2. Semi-supervised: Training data contains ONLY normal points
            X_train_normal = X_train[y_train == 0]
            
            # 3. Feature Scaling (Important for LOF/OCSVM)
            if model_name.startswith('LOF') or model_name.startswith('OCSVM'):
                scaler = StandardScaler()
                X_train_final = scaler.fit_transform(X_train_normal)
                X_test_final = scaler.transform(X_test)
            else:
                X_train_final = X_train_normal
                X_test_final = X_test
            
            # 4. Instantiate and Fit
            start_time = time.time()
            model = model_factory(seed)
            model.fit(X_train_final)
            
            # 5. Predict scores
            # Expect Higher Score = More Anomalous
            if 'IF' in model_name or 'EIF' in model_name or 'SCiF' in model_name or 'FCF' in model_name:
                # isotree scoring must be explicit
                try:
                    scores = model.predict(X_test_final, output="score")
                except (TypeError, ValueError):
                    try:
                        scores = model.decision_function(X_test_final)
                    except (AttributeError, TypeError):
                        scores = model.predict(X_test_final)
                
                # Sanity check: ensure scores are not just binary
                if hasattr(scores, 'ndim') and scores.ndim != 1:
                    scores = scores.flatten()
                
                if np.unique(scores).size <= 10:
                    print(f"Warning: {model_name} returned few unique scores ({np.unique(scores).size}).")
                    
            elif 'LOF' in model_name:
                # decision_function = shifted opposite of anomaly score. 
                # Larger = Inlier. Smaller = Outlier.
                # We want Larger = Outlier. So negate it.
                scores = -model.decision_function(X_test_final)
                
            elif 'OCSVM' in model_name:
                # decision_function: Positive = Inlier, Negative = Outlier.
                # We want Larger = Outlier. So negate it.
                scores = -model.decision_function(X_test_final)
                
            else:
                scores = model.predict(X_test_final)
                
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            # 6. Score
            roc = roc_auc_score(y_test, scores)
            pr = average_precision_score(y_test, scores)
            
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

def compare_models_on_dataset(dataset_name, model_subset=None):
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
        res = test_model(name, factory, X, y)
        if res:
            res['dataset'] = dataset_name
            results.append(res)
            
    return results

if __name__ == "__main__":
    # If run directly, run experiments
    datasets = [
        "arrhythmia",
        "pima",
        "spambase",
        "satellite",
        "pendigits",
        "annthyroid",
        "mnist",
    ]
    
    test_models = [
        'IF', 'IF-U', 'EIF-o', 'EIF-t', 
        'SCiF', 'SCiF-u', 'FCF', 
        'LOF', 'OCSVM-rbf', 'OCSVM-linear'
    ]
    
    # Setup CSV
    if os.path.exists('anomaly_detection_results.csv'):
        os.remove('anomaly_detection_results.csv')
    
    pd.DataFrame(columns=['dataset', 'model', 
                            'roc_auc_mean', 'roc_auc_std', 
                            'pr_auc_mean', 'pr_auc_std', 
                            'train_time']).to_csv('anomaly_detection_results.csv', index=False)
    
    for dataset in datasets:
        try:
            current_models = test_models.copy()
            if dataset == 'covertype':
                # Skip slow models
                for m in ['OCSVM-rbf', 'OCSVM-linear', 'LOF']:
                    if m in current_models:
                        current_models.remove(m)
            
            dataset_results = compare_models_on_dataset(dataset, model_subset=current_models)
            
            if dataset_results:
                df = pd.DataFrame(dataset_results)
                # Enforce column order to match CSV header
                df = df[['dataset', 'model', 'roc_auc_mean', 'roc_auc_std', 'pr_auc_mean', 'pr_auc_std', 'train_time']]
                df.to_csv('anomaly_detection_results.csv', mode='a', header=False, index=False)
                print(f"Saved results for {dataset}")
                
        except Exception as e:
            print(f"\nERROR with dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
