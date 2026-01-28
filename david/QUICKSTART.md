# 🚀 Quick Start Guide

## One-Command Full Pipeline

```bash
python run_all.py
```

This will:
1. ✅ Check dependencies
2. 🔬 Run all experiments
3. 📊 Generate visualizations
4. 📝 Create summary report

## Quick Mode (Faster)

```bash
python run_all.py --quick
```

## Individual Scripts

### Run Experiments Only
```bash
python replicate_paper.py
```

### Analyze Existing Results
```bash
python analyze_results.py
```

### Compare Basic Models
```bash
python compare_models.py
```

### Test Single Model
```bash
python test_datasets.py
```

## 📊 What You Get

### Data Files
- `anomaly_detection_results.csv` - Raw experimental results

### Visualizations
- `comparison_roc_auc.png` - Model comparison (ROC-AUC)
- `comparison_pr_auc.png` - Model comparison (PR-AUC)
- `heatmap_roc_auc.png` - Performance heatmap
- `heatmap_pr_auc.png` - PR-AUC heatmap
- `time_vs_performance.png` - Speed vs accuracy trade-offs

### Documentation
- `README_experiments.md` - Detailed methodology
- `SUMMARY.md` - Complete project summary

## 🔧 Customization

### Add More Datasets

Edit `replicate_paper.py`, line ~565:
```python
datasets = [
    "arrhythmia",
    "pima", 
    "spambase",
    "satellite",    # Uncomment to add
    "pendigits",    # Uncomment to add
    # ... etc
]
```

### Change Models Tested

Edit `replicate_paper.py`, line ~572:
```python
test_models = [
    'IF', 'IF-U', 'EIF-o', 'EIF-t',
    'SCiF', 'SCiF-u', 'FCF', 'DEF',
    'LOF', 'OCSVM-rbf', 'OCSVM-linear'
]
```

### Adjust Model Parameters

Edit `get_model_configs()` in `replicate_paper.py`:
```python
'FCF': {
    'func': lambda: IsolationForest(
        ntrees=200,        # Change number of trees
        sample_size=256,   # Change sample size
        ndim=2,           # Change dimensionality
        # ...
    ),
}
```

## 📈 Key Models Explained

| Model | What It Is | Best For |
|-------|-----------|----------|
| **IF** | Standard Isolation Forest | Fast, general-purpose |
| **EIF-o** | Extended IF (2D hyperplanes) | Better than standard IF |
| **FCF** | Fair-Cut Forest | Multi-modal outliers |
| **DEF** | Density Estimation | Density-based detection |
| **SCiF** | Split Criterion IF | Extreme outliers |
| **LOF** | Local Outlier Factor | Local context important |

## 🎯 Quick Results Interpretation

### Good ROC-AUC Scores
- **>0.90**: Excellent
- **0.80-0.90**: Very good
- **0.70-0.80**: Good
- **<0.70**: Fair/Poor

### Model Selection Tips
1. **Speed matters**: Use standard IF
2. **Accuracy matters**: Use EIF-o or DEF
3. **Multi-modal data**: Use FCF
4. **Research/experiments**: Test all variants

## 🐛 Troubleshooting

### "No module named 'isotree'"
```bash
pip install isotree scikit-learn numpy pandas matplotlib seaborn
```

### "No results file found"
Run experiments first:
```bash
python replicate_paper.py
```

### Experiments take too long
Use quick mode:
```bash
python run_all.py --quick
```

Or reduce datasets in `replicate_paper.py`

### Memory issues
Reduce sample size in model configs or comment out large datasets (MNIST, CoverType)

## 📖 Further Reading

- **Full documentation**: `README_experiments.md`
- **Complete summary**: `SUMMARY.md`
- **Original paper**: https://arxiv.org/pdf/2110.13402

## 💡 Tips

1. Start with quick mode to test setup
2. Run full experiments overnight
3. Analyze results interactively
4. Customize for your specific use case
5. Compare multiple runs for stability

## 🆘 Getting Help

1. Check error messages carefully
2. Review documentation files
3. Ensure all dependencies installed
4. Verify Python 3.7+ installed
5. Check available disk space

## ✨ Advanced Usage

### Run specific model only
Edit `test_models` list to include only desired models

### Custom metrics
Modify `test_model()` function to add additional metrics

### Export to LaTeX
Parse CSV results and format for academic papers

### Statistical testing
Add significance tests between models using scipy

### Cross-validation
Modify train/test split to use k-fold CV

---

**Ready to start? Just run:**
```bash
python run_all.py
```
