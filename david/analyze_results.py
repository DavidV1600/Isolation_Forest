"""
Visualization and Analysis Script for Isolation Forest Experiments

Creates plots and detailed analysis of the experimental results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(csv_file='anomaly_detection_results.csv'):
    """Load results from CSV if available"""
    try:
        df = pd.DataFrame(pd.read_csv(csv_file))
        return df
    except:
        print("No results file found. Run replicate_paper.py first.")
        return None

def plot_model_comparison(df, metric='roc_auc'):
    """Create bar plot comparing models across datasets"""
    
    pivot = df.pivot_table(values=metric, index='model', columns='dataset', aggfunc='first')
    
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title(f'Model Comparison - {metric.upper().replace("_", "-")}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel(f'{metric.upper().replace("_", "-")} Score', fontsize=12)
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'comparison_{metric}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison_{metric}.png")
    plt.close()

def plot_time_vs_performance(df):
    """Plot training time vs performance trade-off"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    datasets = df['dataset'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    
    for idx, dataset in enumerate(datasets):
        subset = df[df['dataset'] == dataset]
        
        # ROC-AUC vs Time
        axes[0].scatter(subset['train_time'], subset['roc_auc'], 
                       s=150, alpha=0.7, color=colors[idx], 
                       label=dataset.capitalize(), edgecolors='black', linewidth=1)
        
        # PR-AUC vs Time  
        axes[1].scatter(subset['train_time'], subset['pr_auc'], 
                       s=150, alpha=0.7, color=colors[idx],
                       label=dataset.capitalize(), edgecolors='black', linewidth=1)
        
        # Annotate model names
        for _, row in subset.iterrows():
            axes[0].annotate(row['model'], 
                           (row['train_time'], row['roc_auc']),
                           fontsize=7, alpha=0.7, 
                           xytext=(3, 3), textcoords='offset points')
            axes[1].annotate(row['model'],
                           (row['train_time'], row['pr_auc']),
                           fontsize=7, alpha=0.7,
                           xytext=(3, 3), textcoords='offset points')
    
    axes[0].set_xlabel('Training Time (seconds)', fontsize=12)
    axes[0].set_ylabel('ROC-AUC', fontsize=12)
    axes[0].set_title('Performance vs Training Time (ROC-AUC)', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    
    axes[1].set_xlabel('Training Time (seconds)', fontsize=12)
    axes[1].set_ylabel('PR-AUC', fontsize=12)
    axes[1].set_title('Performance vs Training Time (PR-AUC)', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('time_vs_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved time_vs_performance.png")
    plt.close()

def plot_heatmap(df, metric='roc_auc'):
    """Create heatmap of model performance across datasets"""
    
    pivot = df.pivot_table(values=metric, index='model', columns='dataset', aggfunc='first')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn', 
                vmin=0.2, vmax=1.0, ax=ax, cbar_kws={'label': metric.upper().replace('_', '-')})
    
    ax.set_title(f'Heatmap: {metric.upper().replace("_", "-")} Scores', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'heatmap_{metric}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap_{metric}.png")
    plt.close()

def generate_ranking_table(df):
    """Generate ranking of models for each dataset"""
    
    print("\n" + "="*100)
    print("MODEL RANKINGS BY DATASET (ROC-AUC)")
    print("="*100)
    
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        subset = df[df['dataset'] == dataset].copy()
        subset = subset.sort_values('roc_auc', ascending=False)
        
        print(f"\n{dataset.upper()}:")
        print(f"{'Rank':<6} {'Model':<15} {'ROC-AUC':<10} {'PR-AUC':<10} {'Time (s)':<12}")
        print("-"*60)
        
        for rank, (_, row) in enumerate(subset.iterrows(), 1):
            print(f"{rank:<6} {row['model']:<15} {row['roc_auc']:<10.4f} "
                  f"{row['pr_auc']:<10.4f} {row['train_time']:<12.5f}")
    
    print("\n" + "="*100)

def generate_summary_statistics(df):
    """Generate summary statistics across all experiments"""
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    # Overall best models
    print("\n🏆 OVERALL BEST MODELS (Average ROC-AUC):")
    avg_roc = df.groupby('model')['roc_auc'].mean().sort_values(ascending=False)
    for model, score in avg_roc.head(5).items():
        print(f"  {model:<15}: {score:.4f}")
    
    # Fastest models
    print("\n⚡ FASTEST MODELS (Average Training Time):")
    avg_time = df.groupby('model')['train_time'].mean().sort_values()
    for model, time_val in avg_time.head(5).items():
        print(f"  {model:<15}: {time_val:.5f}s")
    
    # Best efficiency (ROC-AUC per second)
    print("\n📊 BEST EFFICIENCY (ROC-AUC / Training Time):")
    df['efficiency'] = df['roc_auc'] / df['train_time']
    avg_eff = df.groupby('model')['efficiency'].mean().sort_values(ascending=False)
    for model, eff in avg_eff.head(5).items():
        print(f"  {model:<15}: {eff:.2f}")
    
    # Dataset difficulty (lower average ROC = harder)
    print("\n🎯 DATASET DIFFICULTY (Average ROC-AUC across all models):")
    dataset_diff = df.groupby('dataset')['roc_auc'].mean().sort_values()
    for dataset, score in dataset_diff.items():
        print(f"  {dataset.capitalize():<15}: {score:.4f}")
    
    print("\n" + "="*100)

def analyze_model_families(df):
    """Analyze performance by model family"""
    
    print("\n" + "="*100)
    print("MODEL FAMILY ANALYSIS")
    print("="*100)
    
    # Define families
    families = {
        'Isolation Forest': ['IF', 'IF-U', 'EIF-o', 'EIF-t', 'SCiF', 'SCiF-u', 'FCF', 'DEF'],
        'Distance-based': ['LOF'],
        'SVM-based': ['OCSVM-rbf', 'OCSVM-linear']
    }
    
    for family_name, models in families.items():
        family_df = df[df['model'].isin(models)]
        if len(family_df) > 0:
            avg_roc = family_df['roc_auc'].mean()
            avg_pr = family_df['pr_auc'].mean()
            avg_time = family_df['train_time'].mean()
            
            print(f"\n{family_name}:")
            print(f"  Average ROC-AUC: {avg_roc:.4f}")
            print(f"  Average PR-AUC:  {avg_pr:.4f}")
            print(f"  Average Time:    {avg_time:.5f}s")
            print(f"  Best model:      {family_df.groupby('model')['roc_auc'].mean().idxmax()}")
    
    print("\n" + "="*100)

def main():
    """Main execution function"""
    
    print("="*100)
    print("ISOLATION FOREST EXPERIMENTS - ANALYSIS & VISUALIZATION")
    print("="*100)
    
    # Load results
    df = load_results()
    
    if df is None:
        return
    
    print(f"\nLoaded {len(df)} experimental results")
    print(f"Models tested: {len(df['model'].unique())}")
    print(f"Datasets tested: {len(df['dataset'].unique())}")
    
    # Generate visualizations
    print("\n" + "-"*100)
    print("GENERATING VISUALIZATIONS...")
    print("-"*100)
    
    try:
        plot_model_comparison(df, 'roc_auc')
        plot_model_comparison(df, 'pr_auc')
        plot_heatmap(df, 'roc_auc')
        plot_heatmap(df, 'pr_auc')
        plot_time_vs_performance(df)
        print("\n✓ All visualizations generated successfully!")
    except Exception as e:
        print(f"\n✗ Visualization error: {e}")
    
    # Generate analysis
    print("\n" + "-"*100)
    print("GENERATING ANALYSIS...")
    print("-"*100)
    
    generate_ranking_table(df)
    generate_summary_statistics(df)
    analyze_model_families(df)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100)
    print("\nGenerated files:")
    print("  - comparison_roc_auc.png")
    print("  - comparison_pr_auc.png")
    print("  - heatmap_roc_auc.png")
    print("  - heatmap_pr_auc.png")
    print("  - time_vs_performance.png")
    print("="*100)

if __name__ == "__main__":
    main()
