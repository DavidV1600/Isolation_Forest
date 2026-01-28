import numpy as np
from isotree import IsolationForest
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
import time
import pandas as pd

def load_dataset(dataset_name):
    """Load and prepare different datasets for anomaly detection testing"""
    
    if dataset_name == "arrhythmia":
        print("Loading Arrhythmia dataset...")
        data = fetch_openml('arrhythmia', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Convert to binary: normal (class 1) vs anomaly (other classes)
        y_binary = (y != '1').astype(int)
        
    elif dataset_name == "pima":
        print("Loading Pima (Diabetes) dataset...")
        data = fetch_openml('diabetes', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Convert to binary: 0 = normal, 1 = anomaly
        y_binary = (y == 'tested_positive').astype(int)
        
    elif dataset_name == "spambase":
        print("Loading SpamBase dataset...")
        data = fetch_openml('spambase', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Convert to binary: 0 = normal (not spam), 1 = anomaly (spam)
        y_binary = (y == '1').astype(int)
        
    elif dataset_name == "satellite":
        print("Loading Satellite dataset...")
        data = fetch_openml('satellite', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Convert to binary: smallest classes as anomalies
        from collections import Counter
        class_counts = Counter(y)
        # Use class 2 as normal, others as anomalies
        y_binary = (y != '2').astype(int)
        
    elif dataset_name == "pendigits":
        print("Loading Pendigits dataset...")
        data = fetch_openml('pendigits', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Use digit 0 as normal, others as anomalies
        y_binary = (y != '0').astype(int)
        
    elif dataset_name == "annthyroid":
        print("Loading Annthyroid dataset...")
        data = fetch_openml('annthyroid', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Convert to binary: class 3 = normal, others = anomaly
        y_binary = (y != '3').astype(int)
        
    elif dataset_name == "mnist":
        print("Loading MNIST dataset (this may take a while)...")
        data = fetch_openml('mnist_784', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target.astype(int))
        # Normalize
        X = X / 255.0
        # Use digit 0 as normal, others as anomalies
        y_binary = (y != 0).astype(int)
        
    elif dataset_name == "covertype":
        print("Loading ForestCover (CoverType) dataset...")
        data = fetch_openml('covertype', version=3, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Use class 2 as normal (most common), others as anomalies
        y_binary = (y != '2').astype(int)
        # Sample to make it manageable
        if len(X) > 50000:
            print(f"  Sampling 50000 from {len(X)} samples...")
            idx = np.random.choice(len(X), 50000, replace=False)
            X = X[idx]
            y_binary = y_binary[idx]
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Handle missing values
    if np.any(np.isnan(X)):
        print(f"Found {np.sum(np.isnan(X))} missing values, filling with column means...")
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    
    return X, y_binary

def test_isolation_forest(dataset_name, sample_size=256, ntrees=100):
    """Test Isolation Forest on a specific dataset"""
    
    print("\n" + "="*70)
    print(f"TESTING ISOLATION FOREST ON {dataset_name.upper()} DATASET")
    print("="*70)
    
    # Load dataset
    start_load = time.time()
    X, y = load_dataset(dataset_name)
    load_time = time.time() - start_load
    
    print(f"\nDataset statistics:")
    print(f"  - Shape: {X.shape[0]} rows × {X.shape[1]} columns")
    print(f"  - Anomalies: {np.sum(y)} ({100*np.mean(y):.1f}%)")
    print(f"  - Normal: {np.sum(y==0)} ({100*np.mean(y==0):.1f}%)")
    print(f"  - Load time: {load_time:.2f}s")
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train only on normal data
    X_train_normal = X_train[y_train == 0]
    print(f"\nTraining on {len(X_train_normal)} normal samples...")
    
    # Train Isolation Forest
    start_train = time.time()
    iso = IsolationForest(
        ntrees=ntrees,
        sample_size=min(sample_size, len(X_train_normal)),
        ndim=2,
        nthreads=-1,
        random_seed=42
    )
    iso.fit(X_train_normal)
    train_time = time.time() - start_train
    print(f"Training time: {train_time:.2f}s")
    
    # Predict on test set
    start_pred = time.time()
    anomaly_scores = iso.predict(X_test)
    pred_time = time.time() - start_pred
    print(f"Prediction time: {pred_time:.2f}s")
    
    # Calculate metrics
    # Higher anomaly score = more anomalous
    roc_auc = roc_auc_score(y_test, anomaly_scores)
    pr_auc = average_precision_score(y_test, anomaly_scores)  # PR-AUC
    
    # Convert scores to binary predictions (threshold at median)
    threshold = np.median(anomaly_scores)
    y_pred = (anomaly_scores > threshold).astype(int)
    
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n" + "-"*70)
    print("RESULTS:")
    print("-"*70)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"PR-AUC Score:  {pr_auc:.4f}")
    print(f"Precision:     {precision:.4f}")
    print(f"Recall:        {recall:.4f}")
    print(f"F1-Score:      {f1:.4f}")
    print("-"*70)
    
    return {
        'dataset': dataset_name,
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'anomaly_rate': np.mean(y),
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_time': train_time,
        'pred_time': pred_time
    }

if __name__ == "__main__":
    # Test on multiple datasets (ordered by size for efficiency)
    datasets = [
        "arrhythmia",    # 452 × 279
        "pima",          # 768 × 8
        "spambase",      # 4601 × 57
        "satellite",     # 6435 × 36
        "pendigits",     # 6870 × 16
        "annthyroid",    # 7200 × 6
        "mnist",         # 7603 × 100
        # "covertype",   # Large dataset - uncomment if needed
    ]
    
    all_results = []
    
    for dataset in datasets:
        try:
            result = test_isolation_forest(dataset)
            all_results.append(result)
        except Exception as e:
            print(f"\nError testing {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary table (like Table 6)
    if all_results:
        print("\n\n" + "="*100)
        print("TABLE: Results obtained by Isolation Forest")
        print("="*100)
        print(f"{'Dataset':<15} {'Rows':>8} {'Cols':>5} {'Outliers':>9} {'ROC':>8} {'PR':>8} {'Time':>10}")
        print("-"*100)
        
        for result in all_results:
            dataset_name = result['dataset'].capitalize()
            rows = result['n_samples']
            cols = result['n_features']
            outlier_pct = f"{result['anomaly_rate']*100:.1f}%"
            roc = f"{result['roc_auc']:.4f}"
            pr = f"{result['pr_auc']:.4f}"
            time_val = f"{result['train_time']:.4f}"
            
            print(f"{dataset_name:<15} {rows:>8} {cols:>5} {outlier_pct:>9} {roc:>8} {pr:>8} {time_val:>10}")
        
        print("="*100)
        
        # Additional detailed summary
        print("\n\nDETAILED SUMMARY:")
        print("-"*100)
        for result in all_results:
            print(f"\n{result['dataset'].upper()}:")
            print(f"  Shape: {result['n_samples']} × {result['n_features']}")
            print(f"  Outliers: {result['anomaly_rate']*100:.1f}%")
            print(f"  ROC-AUC: {result['roc_auc']:.4f}")
            print(f"  PR-AUC: {result['pr_auc']:.4f}")
            print(f"  F1-Score: {result['f1_score']:.4f}")
            print(f"  Train time: {result['train_time']:.4f}s")
            print(f"  Pred time: {result['pred_time']:.4f}s")
