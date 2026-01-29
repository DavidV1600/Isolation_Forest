import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import time

# Import multiple anomaly detection models
from isotree import IsolationForest as IsoTree_IF
from sklearn.ensemble import IsolationForest as SKLearn_IF
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope

def load_dataset(dataset_name):
    """Load and prepare different datasets for anomaly detection testing"""
    
    if dataset_name == "arrhythmia":
        data = fetch_openml('arrhythmia', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        y_binary = (y != '1').astype(int)
        
    elif dataset_name == "pima":
        data = fetch_openml('diabetes', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        y_binary = (y == 'tested_positive').astype(int)
        
    elif dataset_name == "spambase":
        data = fetch_openml('spambase', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        y_binary = (y == '1').astype(int)
        
    elif dataset_name == "pendigits":
        data = fetch_openml('pendigits', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target)
        y_binary = (y != '0').astype(int)
        
    elif dataset_name == "mnist":
        data = fetch_openml('mnist_784', version=1, parser='auto')
        X = np.array(data.data, dtype=np.float32)
        y = np.array(data.target.astype(int))
        X = X / 255.0
        y_binary = (y != 0).astype(int)
        # Sample to reduce size
        if len(X) > 10000:
            idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
            X = X[idx]
            y_binary = y_binary[idx]
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Handle missing values
    if np.any(np.isnan(X)):
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    
    return X, y_binary

def get_models():
    """Return dictionary of anomaly detection models"""
    models = {
        'IF-isotree': lambda: IsoTree_IF(
            ntrees=100,
            sample_size=256,
            ndim=1,
            nthreads=-1,
            random_seed=42
        ),
        'IF-sklearn': lambda: SKLearn_IF(
            n_estimators=100,
            max_samples=256,
            contamination='auto',
            random_state=42,
            n_jobs=-1
        ),
        'EIF': lambda: IsoTree_IF(
            ntrees=100,
            sample_size=256,
            ndim=2,  # Extended IF uses 2D splits
            nthreads=-1,
            random_seed=42
        ),
        'LOF': lambda: LocalOutlierFactor(
            n_neighbors=20,
            contamination='auto',
            novelty=True,
            n_jobs=-1
        ),
        'OCSVM-rbf': lambda: OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.1
        ),
        'OCSVM-linear': lambda: OneClassSVM(
            kernel='linear',
            nu=0.1
        ),
    }
    return models

def test_model(model_name, model_func, X_train_normal, X_test, y_test):
    """Test a single model and return metrics"""
    try:
        print(f"  Testing {model_name}...", end=' ')
        
        # Train
        start_train = time.time()
        model = model_func()
        model.fit(X_train_normal)
        train_time = time.time() - start_train
        
        # Predict
        start_pred = time.time()
        if hasattr(model, 'predict'):
            if model_name.startswith('IF') or model_name == 'EIF':
                # isotree scoring must be explicit
                try:
                    anomaly_scores = model.predict(X_test, output="score")
                except (TypeError, ValueError):
                    try:
                        anomaly_scores = model.decision_function(X_test)
                    except (AttributeError, TypeError):
                        anomaly_scores = model.predict(X_test)
                
                # Sanity check
                if hasattr(anomaly_scores, 'ndim') and anomaly_scores.ndim != 1:
                    anomaly_scores = anomaly_scores.flatten()
            else:
                # Other models (LOF, OCSVM) return -1/1, convert to scores
                predictions = model.predict(X_test)
                if hasattr(model, 'decision_function'):
                    anomaly_scores = -model.decision_function(X_test)
                else:
                    anomaly_scores = -predictions
        pred_time = time.time() - start_pred
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_test, anomaly_scores)
            pr_auc = average_precision_score(y_test, anomaly_scores)
        except:
            # If scores are not proper, try negating or using predictions
            roc_auc = roc_auc_score(y_test, -anomaly_scores)
            pr_auc = average_precision_score(y_test, -anomaly_scores)
        
        print(f"ROC: {roc_auc:.4f}, PR: {pr_auc:.4f}, Time: {train_time:.4f}s")
        
        return {
            'model': model_name,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'train_time': train_time,
            'pred_time': pred_time,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"FAILED - {str(e)[:50]}")
        return {
            'model': model_name,
            'roc_auc': np.nan,
            'pr_auc': np.nan,
            'train_time': np.nan,
            'pred_time': np.nan,
            'status': f'failed: {str(e)[:50]}'
        }

def compare_models_on_dataset(dataset_name):
    """Compare all models on a single dataset"""
    
    print("\n" + "="*80)
    print(f"DATASET: {dataset_name.upper()}")
    print("="*80)
    
    # Load dataset
    print("Loading dataset...", end=' ')
    X, y = load_dataset(dataset_name)
    print(f"Done. Shape: {X.shape[0]} × {X.shape[1]}, Outliers: {np.mean(y)*100:.1f}%")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train only on normal data
    X_train_normal = X_train[y_train == 0]
    print(f"Training samples (normal only): {len(X_train_normal)}")
    print(f"Test samples: {len(X_test)}")
    
    # Get all models
    models = get_models()
    
    # Test each model
    results = []
    for model_name, model_func in models.items():
        result = test_model(model_name, model_func, X_train_normal, X_test, y_test)
        result['dataset'] = dataset_name
        results.append(result)
    
    return results

if __name__ == "__main__":
    # Test on multiple datasets
    datasets = [
        "arrhythmia",
        "pima",
        "spambase",
        "pendigits",
        "mnist",
    ]
    
    all_results = []
    
    for dataset in datasets:
        try:
            results = compare_models_on_dataset(dataset)
            all_results.extend(results)
        except Exception as e:
            print(f"\nError with dataset {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # Convert to DataFrame for better visualization
    df = pd.DataFrame(all_results)
    
    # Create comparison table (like Table 6)
    print("\n\n" + "="*120)
    print("TABLE: Results obtained by each method")
    print("="*120)
    
    # Pivot table: rows=models, columns=datasets with ROC/PR pairs
    for metric in ['roc_auc', 'pr_auc']:
        print(f"\n{metric.upper()}:")
        pivot = df[df['status'] == 'success'].pivot_table(
            values=metric,
            index='model',
            columns='dataset',
            aggfunc='first'
        )
        print(pivot.to_string())
    
    # Summary with time
    print("\n\nTRAINING TIME (seconds):")
    pivot_time = df[df['status'] == 'success'].pivot_table(
        values='train_time',
        index='model',
        columns='dataset',
        aggfunc='first'
    )
    print(pivot_time.to_string())
    
    # Best model per dataset
    print("\n\n" + "="*120)
    print("BEST MODEL PER DATASET (by ROC-AUC):")
    print("="*120)
    df_success = df[df['status'] == 'success']
    for dataset in datasets:
        dataset_results = df_success[df_success['dataset'] == dataset]
        if len(dataset_results) > 0:
            best = dataset_results.loc[dataset_results['roc_auc'].idxmax()]
            print(f"{dataset:15s}: {best['model']:15s} ROC={best['roc_auc']:.4f}, PR={best['pr_auc']:.4f}")
    
    print("\n" + "="*120)
