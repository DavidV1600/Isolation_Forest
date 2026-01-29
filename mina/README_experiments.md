# Isolation Forest Experiments - Paper Replication

This directory contains scripts to replicate the experiments from:
**"Revisiting randomized choices in isolation forests"** by David Cortes  
Paper: https://arxiv.org/pdf/2110.13402

## 📁 Files Overview

### 1. `test_datasets.py`
Basic testing script that evaluates a single Isolation Forest model on multiple datasets.
- Tests datasets: Arrhythmia, Pima, SpamBase, Pendigits, Annthyroid, MNIST
- Reports: ROC-AUC, PR-AUC, training time
- Simple and fast for quick benchmarking

### 2. `compare_models.py`
Comparative analysis of multiple anomaly detection algorithms.
- **Models tested:**
  - IF-isotree (standard Isolation Forest)
  - IF-sklearn (scikit-learn's implementation)
  - EIF (Extended Isolation Forest)
  - LOF (Local Outlier Factor)
  - OCSVM-rbf (One-Class SVM with RBF kernel)
  - OCSVM-linear (One-Class SVM with linear kernel)
- Generates pivot tables comparing all models across datasets
- Identifies best model per dataset

### 3. `replicate_paper.py` ⭐ **MAIN SCRIPT**
Complete replication of the paper's experiments with all isolation forest variants.

#### Models Implemented:
| Model | Description | Key Feature |
|-------|-------------|-------------|
| **IF** | Standard Isolation Forest | Uniform random splits, axis-parallel |
| **IF-U** | IF with non-uniform var selection | Uses pooled gain for variable selection |
| **EIF-o** | Extended IF - Original | 2D random hyperplane splits |
| **EIF-t** | Extended IF - Tweaked | EIF + pooled gain variable selection |
| **SCiF** | Split Criterion IF | Averaged gain criterion, tries 10 splits |
| **SCiF-u** | SCiF with uniform var selection | SCiF + pooled gain |
| **FCF** | Fair-Cut Forest | **Paper's main contribution**, pooled gain, 200 trees |
| **DEF** | Density Estimation Forest | 3D splits with pooled gain |
| **LOF** | Local Outlier Factor | Density-based comparison method |
| **OCSVM** | One-Class SVM | Support vector baseline |

#### Key Parameters (from paper):
- **Standard IF**: 100 trees, 256 samples, 1D axis-parallel splits
- **FCF**: 200 trees, 256 samples, 2D splits, pooled gain criterion
- **SCiF**: 100 trees, 256 samples, 2D splits, 10 trials, averaged gain
- **EIF**: 100 trees, 256 samples, 2D random hyperplane splits

## 🎯 Datasets (Table 5 from paper)

| Dataset | Rows | Columns | Outliers | Type |
|---------|------|---------|----------|------|
| Arrhythmia | 452 | 274 | 15% | Medical, multi-modal |
| Pima | 768 | 8 | 35% | Medical, binary classification |
| SpamBase | 4,601 | 57 | 39.4% | Text, skewed distributions |
| Satellite | 6,435 | 36 | 32% | Remote sensing, multi-modal |
| Pendigits | 6,870 | 16 | 2.27% | Handwriting, low anomaly rate |
| Annthyroid | 7,200 | 6 | 7.42% | Medical, opposite extremes |
| MNIST | 7,603 | 100 | 9.2% | Images, downsampled |
| ALOI | 50,000 | 27* | 3% | Images, rare objects |
| ForestCover | 286,048 | 10 | 0.9% | Geographical, large dataset |

*Contains categorical variable

## 🚀 Usage

### Quick Test (Single Model)
```bash
python test_datasets.py
```

### Compare Multiple Algorithms
```bash
python compare_models.py
```

### Full Paper Replication
```bash
python replicate_paper.py
```

## 📊 Key Findings

### Current Results (3 datasets tested):

#### Arrhythmia (452 × 279, 45.8% outliers)
- **Best**: LOF (0.8041 ROC-AUC)
- EIF-o: 0.7936
- IF: 0.7833
- DEF: 0.7768
- FCF: 0.7574

#### Pima (768 × 8, 34.9% outliers)
- **Best**: DEF (0.7453 ROC-AUC)
- FCF: 0.7362
- EIF-t: 0.7328
- IF-U: 0.7094
- IF: 0.6877

#### SpamBase (4,601 × 57, 39.4% outliers)
- **Best**: EIF-o (0.8672 ROC-AUC)
- DEF: 0.8618
- EIF-t: 0.8553
- SCiF-u: 0.8503
- FCF: 0.8454
- IF: 0.8443

### Speed Comparison:
- **Fastest**: IF variants (0.002-0.008s)
- **Medium**: FCF, DEF (0.004-0.011s)
- **Slower**: SCiF variants (0.028-0.071s) - due to 10 trials
- **Slowest**: LOF, OCSVM (0.04-0.23s)

## 📖 Paper Insights

### Types of Outliers:
1. **Scattered outliers**: Random extreme values (Pima, ForestCover, MNIST)
2. **Local outliers**: Near a single minority mode (ALOI)
3. **Clustered outliers**: Multiple minority modes (Satellite, Arrhythmia, Annthyroid)

### Key Concepts:

**Pooled Gain Criterion**:
- Aims to split data evenly in early branches
- Tends to separate modes effectively
- Better for multi-modal datasets
- Expected depth for uniform data: E[d(m)] = log₂(m)

**Averaged Gain Criterion**:
- Quickly isolates extreme values
- Favors splits with few points in one branch
- Can create "ghost regions" in some cases
- Expected depth formula: E[d(m)] = (Σᵢ₌₁ᵐ i)/m - 1

**Extended Isolation Forest (EIF)**:
- Uses random hyperplane splits instead of axis-parallel
- Reduces bias from axis-aligned splits
- Better handles distributions at angles to axes

**Fair-Cut Forest (FCF)**:
- Paper's main contribution
- Combines 2D splits with pooled gain
- Uses 200 trees for convergence
- Optimized for clustered multi-modal outliers

## 🔬 Extensions & Future Work

### To Add:
1. ✅ RRCF (Robust Random Cut Forest) - variable selection by range
2. ✅ DET (Density Estimation Trees) 
3. ✅ OCRF (One-Class Random Forest)
4. ✅ GIF (Generalized Isolation Forest)
5. Additional datasets (Satellite, ALOI, larger MNIST, ForestCover)
6. Hyperparameter sensitivity analysis
7. Ghost regions visualization
8. Contamination rate experiments

### Hyperparameter Variations:
- Sample size: 32-512
- Number of trees: 50-500
- ndim: 1-5
- ntry: 1-20
- max_depth effects

## 📝 Notes

- All models trained on **normal data only** (semi-supervised)
- 70-30 train-test split (stratified)
- Missing values imputed with column means
- Metrics: ROC-AUC (primary), PR-AUC (secondary)
- Random seed: 42 for reproducibility

## 🛠️ Dependencies

```bash
pip install isotree scikit-learn numpy pandas
```

## 📚 References

1. **Cortes, D.** (2021). "Revisiting randomized choices in isolation forests". arXiv preprint arXiv:2110.13402.
2. **Liu, F. T., Ting, K. M., & Zhou, Z. H.** (2008). "Isolation forest". In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE.
3. **Hariri, S., Kind, M. C., & Brunner, R. J.** (2019). "Extended isolation forest". IEEE Transactions on Knowledge and Data Engineering.
4. **Liu, F. T., Ting, K. M., & Zhou, Z. H.** (2010). "On detecting clustered anomalies using SCiForest". In Joint European Conference on Machine Learning and Knowledge Discovery in Databases (pp. 274-290). Springer.

---

**Author**: David Cortes  
**Date**: January 2026  
**Based on**: arXiv:2110.13402
