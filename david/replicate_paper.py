"""
Enhanced Anomaly Detection Comparison Script
Replicating experiments from: "Revisiting randomized choices in isolation forests"
https://arxiv.org/pdf/2110.13402

Tests multiple isolation forest variants and competing methods on benchmark datasets.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

# Import anomaly detection models
from isotree import IsolationForest
from sklearn.ensemble import IsolationForest as SKLearn_IF
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def load_dataset(dataset_name):
    """
    Load and prepare datasets used in the paper.
    Returns X (features), y (binary labels: 0=normal, 1=anomaly)
    """
    
    if dataset_name == "arrhythmia":
        data = fetch_openml('arrhythmia', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Convert to binary: class 1 = normal, others = anomaly
        y_binary = (y != '1').astype(int)
        
    elif dataset_name == "pima":
        data = fetch_openml('diabetes', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Binary classification: tested_positive = anomaly
        y_binary = (y == 'tested_positive').astype(int)
        
    elif dataset_name == "spambase":
        data = fetch_openml('spambase', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Spam = anomaly
        y_binary = (y == '1').astype(int)
        
    elif dataset_name == "satellite":
        data = fetch_openml('satellite_image', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Mix frequent and uncommon classes (multi-modal outliers)
        # Classes 1,2,3,5 are normal, 4,6,7 are anomalies
        normal_classes = ['1', '2', '3', '5']
        y_binary = (~np.isin(y, normal_classes)).astype(int)
        
    elif dataset_name == "pendigits":
        data = fetch_openml('pendigits', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Use digit 0 as normal, others as anomalies
        y_binary = (y != '0').astype(int)
        
    elif dataset_name == "annthyroid":
        data = fetch_openml('thyroid-dis', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Class 3 = normal, classes 1,2 = anomalies (low/high)
        y_binary = (y != '3').astype(int)
        
    elif dataset_name == "mnist":
        data = fetch_openml('mnist_784', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target.astype(int))
        X = X / 255.0
        # Use majority digit as normal, sample minority as anomaly
        # Keep only digits 0 (normal) and 6 (anomaly) for a 9.2% anomaly rate
        mask = (y == 0) | (y == 6)
        X = X[mask]
        y = y[mask]
        y_binary = (y == 6).astype(int)
        # Sample to get ~7603 rows with ~9.2% anomalies
        n_normal = 6900
        n_anomaly = 703
        idx_normal = np.where(y_binary == 0)[0][:n_normal]
        idx_anomaly = np.where(y_binary == 1)[0][:n_anomaly]
        idx = np.concatenate([idx_normal, idx_anomaly])
        X = X[idx]
        y_binary = y_binary[idx]
        
    elif dataset_name == "covertype":
        data = fetch_openml('covertype', version=3, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        # Use class 2 as normal, class 4 as anomaly
        mask = (y == '2') | (y == '4')
        X = X[mask]
        y_binary = (y[mask] == '4').astype(int)
        # Sample to manageable size while maintaining 0.9% anomaly rate
        if len(X) > 50000:
            n_samples = 50000
            idx = np.random.RandomState(42).choice(len(X), n_samples, replace=False)
            X = X[idx]
            y_binary = y_binary[idx]
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Handle missing values (column mean imputation)
    if np.any(np.isnan(X)):
        print(f"  Found {np.sum(np.isnan(X))} missing values, imputing with column means...")
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    
    return X, y_binary

def get_model_configs():
    """
    Return dictionary of all model configurations to test.
    Based on the paper's methodology.
    """
    models = {
        # Standard Isolation Forest (IF)
        'IF': {
            'func': lambda: IsolationForest(
                ntrees=100,
                sample_size=256,
                ndim=1,
                prob_pick_pooled_gain=0.0,
                prob_pick_avg_gain=0.0,
                nthreads=-1,
                random_seed=42
            ),
            'description': 'Standard Isolation Forest (uniform random splits)'
        },
        
        # IF with uniform-like variable selection but better (IF-u)
        'IF-U': {
            'func': lambda: IsolationForest(
                ntrees=100,
                sample_size=256,
                ndim=1,
                prob_pick_pooled_gain=1.0,  # Use pooled gain for variable selection
                prob_pick_avg_gain=0.0,
                nthreads=-1,
                random_seed=42
            ),
            'description': 'IF with non-uniform variable selection'
        },
        
        # Extended Isolation Forest - Original (EIF-o)
        'EIF-o': {
            'func': lambda: IsolationForest(
                ntrees=100,
                sample_size=256,
                ndim=2,  # Use 2D hyperplane splits
                prob_pick_avg_gain=0.0,
                prob_pick_pooled_gain=0.0,
                nthreads=-1,
                random_seed=42
            ),
            'description': 'Extended IF - Original (2D random hyperplanes)'
        },
        
        # Extended Isolation Forest - Tweaked (EIF-t)
        'EIF-t': {
            'func': lambda: IsolationForest(
                ntrees=100,
                sample_size=256,
                ndim=2,
                prob_pick_pooled_gain=1.0,
                prob_pick_avg_gain=0.0,
                nthreads=-1,
                random_seed=42
            ),
            'description': 'Extended IF - Tweaked with pooled gain'
        },
        
        # Split Criterion Isolation Forest (SCiF)
        'SCiF': {
            'func': lambda: IsolationForest(
                ntrees=100,
                sample_size=256,
                ndim=2,
                ntry=10,  # Try 10 random splits, choose best by averaged gain
                prob_pick_avg_gain=1.0,
                prob_pick_pooled_gain=0.0,
                nthreads=-1,
                random_seed=42
            ),
            'description': 'Split Criterion IF (averaged gain criterion)'
        },
        
        # SCiF with uniform variable selection (SCiF-u)
        'SCiF-u': {
            'func': lambda: IsolationForest(
                ntrees=100,
                sample_size=256,
                ndim=2,
                ntry=10,
                prob_pick_avg_gain=1.0,
                prob_pick_pooled_gain=1.0,
                nthreads=-1,
                random_seed=42
            ),
            'description': 'SCiF with uniform variable selection'
        },
        
        # Fair-Cut Forest (FCF) - The paper's main contribution
        'FCF': {
            'func': lambda: IsolationForest(
                ntrees=200,  # Paper suggests 200 trees for FCF
                sample_size=256,
                ndim=2,
                ntry=1,  # Only 1 trial per node
                prob_pick_pooled_gain=1.0,  # Use pooled gain
                prob_pick_avg_gain=0.0,
                nthreads=-1,
                random_seed=42
            ),
            'description': 'Fair-Cut Forest (pooled gain, 200 trees)'
        },
        
        # Density Estimation Forest (DEF) approximation
        'DEF': {
            'func': lambda: IsolationForest(
                ntrees=100,
                sample_size=256,
                ndim=3,  # More variables per split
                prob_pick_pooled_gain=1.0,
                prob_pick_avg_gain=0.0,
                nthreads=-1,
                random_seed=42
            ),
            'description': 'Density Estimation Forest approximation'
        },
        
        # Local Outlier Factor
        'LOF': {
            'func': lambda: LocalOutlierFactor(
                n_neighbors=20,
                contamination='auto',
                novelty=True,
                n_jobs=-1
            ),
            'description': 'Local Outlier Factor'
        },
        
        # One-Class SVM with RBF kernel
        'OCSVM-rbf': {
            'func': lambda: OneClassSVM(
                kernel='rbf',
                gamma='auto',
                nu=0.1
            ),
            'description': 'One-Class SVM with RBF kernel'
        },
        
        # One-Class SVM with linear kernel
        'OCSVM-linear': {
            'func': lambda: OneClassSVM(
                kernel='linear',
                nu=0.1
            ),
            'description': 'One-Class SVM with linear kernel'
        },
    }
    
    return models

def test_model(model_name, model_func, X_train_normal, X_test, y_test, timeout=300):
    """Test a single model and return metrics"""
    try:
        print(f"  {model_name:12s}", end=' ')
        
        # Train
        start_train = time.time()
        model = model_func()
        model.fit(X_train_normal)
        train_time = time.time() - start_train
        
        if train_time > timeout:
            print(f"TIMEOUT (>{timeout}s)")
            return None
        
        # Predict
        start_pred = time.time()
        if hasattr(model, 'predict'):
            if 'IF' in model_name or 'EIF' in model_name or 'SCiF' in model_name or 'FCF' in model_name or 'DEF' in model_name:
                # Isolation Forest variants return anomaly scores directly
                anomaly_scores = model.predict(X_test)
            else:
                # LOF, OCSVM return -1/1 predictions
                if hasattr(model, 'decision_function'):
                    anomaly_scores = -model.decision_function(X_test)
                else:
                    predictions = model.predict(X_test)
                    anomaly_scores = -predictions
        pred_time = time.time() - start_pred
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_test, anomaly_scores)
            pr_auc = average_precision_score(y_test, anomaly_scores)
        except:
            # Try negating if needed
            try:
                roc_auc = roc_auc_score(y_test, -anomaly_scores)
                pr_auc = average_precision_score(y_test, -anomaly_scores)
            except:
                print(f"FAILED (metric calculation)")
                return None
        
        print(f"ROC: {roc_auc:.4f}, PR: {pr_auc:.4f}, Time: {train_time:.5f}s")
        
        return {
            'model': model_name,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'train_time': train_time,
            'pred_time': pred_time,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"FAILED ({str(e)[:40]})")
        return None

def compare_models_on_dataset(dataset_name, model_subset=None):
    """
    Compare all models on a single dataset.
    
    Args:
        dataset_name: Name of dataset to test
        model_subset: List of model names to test (None = all models)
    """
    
    print("\n" + "="*90)
    print(f"DATASET: {dataset_name.upper()}")
    print("="*90)
    
    # Load dataset
    print("Loading dataset...", end=' ')
    start_load = time.time()
    X, y = load_dataset(dataset_name)
    load_time = time.time() - start_load
    
    print(f"Done ({load_time:.2f}s)")
    print(f"  Shape: {X.shape[0]} rows × {X.shape[1]} columns")
    print(f"  Outliers: {np.sum(y)} ({100*np.mean(y):.1f}%)")
    print(f"  Normal: {np.sum(y==0)} ({100*np.mean(y==0):.1f}%)")
    
    # Split into train and test (70-30 split as in paper)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train only on normal data (semi-supervised anomaly detection)
    X_train_normal = X_train[y_train == 0]
    print(f"  Training samples (normal only): {len(X_train_normal)}")
    print(f"  Test samples: {len(X_test)} ({np.sum(y_test)} anomalies)")
    
    # Get all models
    all_models = get_model_configs()
    
    # Filter to subset if specified
    if model_subset:
        models = {k: v for k, v in all_models.items() if k in model_subset}
    else:
        models = all_models
    
    print(f"\nTesting {len(models)} models...")
    print("-"*90)
    
    # Test each model
    results = []
    for model_name, config in models.items():
        result = test_model(
            model_name, 
            config['func'], 
            X_train_normal, 
            X_test, 
            y_test
        )
        if result:
            result['dataset'] = dataset_name
            result['n_samples'] = X.shape[0]
            result['n_features'] = X.shape[1]
            result['outlier_rate'] = np.mean(y)
            results.append(result)
    
    return results

def create_paper_table(df):
    """Create Table 6 style results table from the paper"""
    
    print("\n" + "="*120)
    print("TABLE: Results obtained by each method (ROC-AUC, PR-AUC, Time)")
    print("="*120)
    
    # Group by dataset
    datasets = df['dataset'].unique()
    models = df['model'].unique()
    
    # Create formatted output
    header = f"{'Model':<12s}"
    for dataset in datasets:
        header += f" | {dataset.capitalize():^25s}"
    print(header)
    print("-"*120)
    
    for model in models:
        row = f"{model:<12s}"
        for dataset in datasets:
            subset = df[(df['model'] == model) & (df['dataset'] == dataset)]
            if len(subset) > 0:
                roc = subset.iloc[0]['roc_auc']
                pr = subset.iloc[0]['pr_auc']
                time_val = subset.iloc[0]['train_time']
                row += f" | {roc:.4f} {pr:.4f} {time_val:>7.4f}"
            else:
                row += f" | {'---':^25s}"
        print(row)
    
    print("="*120)
    
    # Best model per dataset
    print("\n" + "="*120)
    print("BEST MODEL PER DATASET (by ROC-AUC)")
    print("="*120)
    for dataset in datasets:
        dataset_results = df[df['dataset'] == dataset]
        if len(dataset_results) > 0:
            best = dataset_results.loc[dataset_results['roc_auc'].idxmax()]
            print(f"{dataset.capitalize():15s}: {best['model']:12s} "
                  f"ROC={best['roc_auc']:.4f}, PR={best['pr_auc']:.4f}, "
                  f"Time={best['train_time']:.4f}s")
    print("="*120)

if __name__ == "__main__":
    # Datasets from Table 5 in the paper
    datasets = [
        "arrhythmia",    # 452 × 274, 15% outliers
        "pima",          # 768 × 8, 35% outliers
        "spambase",      # 4601 × 57, 39.4% outliers
        # "satellite",   # 6435 × 36, 32% outliers (requires special class mapping)
        # "pendigits",   # 6870 × 16, 2.27% outliers
        # "annthyroid",  # 7200 × 6, 7.42% outliers
        # "mnist",       # 7603 × 100, 9.2% outliers
        # "covertype",   # Large dataset
    ]
    
    # Models to test (comment out to test all)
    test_models = [
        'IF', 'IF-U', 'EIF-o', 'EIF-t', 
        'SCiF', 'SCiF-u', 'FCF', 'DEF',
        'LOF', 'OCSVM-rbf', 'OCSVM-linear'
    ]
    
    all_results = []
    
    for dataset in datasets:
        try:
            results = compare_models_on_dataset(dataset, model_subset=test_models)
            all_results.extend(results)
        except Exception as e:
            print(f"\nERROR with dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Convert to DataFrame and create comparison table
    if all_results:
        df = pd.DataFrame(all_results)
        create_paper_table(df)
        
        # Save results to CSV
        df.to_csv('anomaly_detection_results.csv', index=False)
        print("\n✓ Results saved to anomaly_detection_results.csv")
