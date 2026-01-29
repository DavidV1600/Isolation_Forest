import numpy as np
from isotree import IsolationForest
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

### Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target.astype(int)

### Convert to numpy arrays if needed
X = np.array(X)
y = np.array(y)

### Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

### Normalize the data (0-255 to 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

### Store results for all models
results = []

print("\n" + "="*70)
print("TRAINING 10 ISOLATION FOREST MODELS (ONE PER DIGIT)")
print("="*70)

### Train a model for each digit (0-9)
for target_digit in range(10):
    print(f"\n{'='*70}")
    print(f"MODEL {target_digit + 1}/10: Training on digit {target_digit} as NORMAL")
    print(f"{'='*70}")
    
    ### Filter training data: only current digit is "normal"
    X_train_normal = X_train[y_train == target_digit]
    print(f"Training samples (digit {target_digit} only): {len(X_train_normal)}")
    
    ### Fit Isolation Forest on normal data
    print(f"Training Isolation Forest for digit {target_digit}...")
    iso = IsolationForest(
        ntrees=100,
        sample_size=256,
        ndim=2,
        nthreads=-1,
        random_seed=42
    )
    iso.fit(X_train_normal)
    
    ### Predict anomaly scores on test set
    print(f"Predicting on test set...")
    # isotree scoring must be explicit
    try:
        anomaly_scores = iso.predict(X_test, output="score")
    except (TypeError, ValueError):
        try:
            anomaly_scores = iso.decision_function(X_test)
        except (AttributeError, TypeError):
            anomaly_scores = iso.predict(X_test)

    # Sanity check
    if hasattr(anomaly_scores, 'ndim') and anomaly_scores.ndim != 1:
        anomaly_scores = anomaly_scores.flatten()
    
    ### Create binary labels: 0 = normal (current digit), 1 = anomaly (other digits)
    y_test_binary = (y_test != target_digit).astype(int)
    
    ### Calculate ROC-AUC using anomaly scores
    roc_auc = roc_auc_score(y_test_binary, anomaly_scores)
    
    ### Store results
    results.append({
        'digit': target_digit,
        'train_samples': len(X_train_normal),
        'test_normal': np.sum(y_test_binary == 0),
        'test_anomaly': np.sum(y_test_binary == 1),
        'roc_auc': roc_auc,
        'mean_score_normal': np.mean(anomaly_scores[y_test_binary == 0]),
        'mean_score_anomaly': np.mean(anomaly_scores[y_test_binary == 1])
    })
    
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Mean anomaly score for digit {target_digit}: {results[-1]['mean_score_normal']:.4f}")
    print(f"Mean anomaly score for other digits: {results[-1]['mean_score_anomaly']:.4f}")

### Display summary of all models
print("\n" + "="*70)
print("SUMMARY OF ALL 10 MODELS")
print("="*70)
print(f"\n{'Digit':<8} {'Train Samples':<15} {'Test Normal':<13} {'Test Anomaly':<13} {'ROC-AUC':<10}")
print("-" * 70)

for res in results:
    print(f"{res['digit']:<8} {res['train_samples']:<15} {res['test_normal']:<13} "
          f"{res['test_anomaly']:<13} {res['roc_auc']:<10.4f}")

### Overall statistics
print("\n" + "="*70)
print("OVERALL STATISTICS")
print("="*70)
avg_roc_auc = np.mean([r['roc_auc'] for r in results])
min_roc_auc = np.min([r['roc_auc'] for r in results])
max_roc_auc = np.max([r['roc_auc'] for r in results])
best_digit = results[np.argmax([r['roc_auc'] for r in results])]['digit']
worst_digit = results[np.argmin([r['roc_auc'] for r in results])]['digit']

print(f"\nAverage ROC-AUC across all digits: {avg_roc_auc:.4f}")
print(f"Best performing digit: {best_digit} (ROC-AUC: {max_roc_auc:.4f})")
print(f"Worst performing digit: {worst_digit} (ROC-AUC: {min_roc_auc:.4f})")
print(f"ROC-AUC Range: {min_roc_auc:.4f} - {max_roc_auc:.4f}")

### Show separation analysis
print("\n" + "="*70)
print("ANOMALY SCORE SEPARATION (Mean Score: Anomaly - Normal)")
print("="*70)
for res in results:
    separation = res['mean_score_anomaly'] - res['mean_score_normal']
    print(f"Digit {res['digit']}: {separation:+.4f} "
          f"(Normal: {res['mean_score_normal']:.4f}, Anomaly: {res['mean_score_anomaly']:.4f})")