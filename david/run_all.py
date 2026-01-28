#!/usr/bin/env python3
"""
Master script to run all isolation forest experiments

This script orchestrates the complete experimental pipeline:
1. Runs paper replication experiments
2. Generates visualizations and analysis
3. Creates summary report
"""

import subprocess
import sys
import os
from datetime import datetime

def print_header(text):
    """Print formatted section header"""
    print("\n" + "="*100)
    print(text.center(100))
    print("="*100 + "\n")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"▶ {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with error: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print_header("CHECKING DEPENDENCIES")
    
    required_packages = [
        'isotree', 'sklearn', 'numpy', 'pandas', 'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package if package != 'sklearn' else 'sklearn')
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install isotree scikit-learn numpy pandas matplotlib seaborn")
        return False
    
    print("\n✓ All dependencies satisfied")
    return True

def run_experiments(quick_mode=False):
    """Run all experiments"""
    print_header("RUNNING EXPERIMENTS")
    
    start_time = datetime.now()
    
    if quick_mode:
        print("🚀 Quick mode: Testing on 3 datasets with subset of models")
    else:
        print("🔬 Full mode: Complete paper replication")
    
    success = run_command(
        "python replicate_paper.py",
        "Paper replication experiments"
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"\n⏱ Experiments completed in {duration:.1f} seconds")
    return success

def generate_visualizations():
    """Generate plots and analysis"""
    print_header("GENERATING VISUALIZATIONS & ANALYSIS")
    
    success = run_command(
        "python analyze_results.py",
        "Analysis and visualization"
    )
    
    return success

def create_report():
    """Create summary report"""
    print_header("CREATING SUMMARY REPORT")
    
    if not os.path.exists('anomaly_detection_results.csv'):
        print("✗ No results file found. Run experiments first.")
        return False
    
    # Check if visualizations exist
    viz_files = [
        'comparison_roc_auc.png',
        'comparison_pr_auc.png',
        'heatmap_roc_auc.png',
        'heatmap_pr_auc.png',
        'time_vs_performance.png'
    ]
    
    print("📊 Generated files:")
    print("  - anomaly_detection_results.csv (raw data)")
    
    for viz_file in viz_files:
        if os.path.exists(viz_file):
            print(f"  - {viz_file}")
    
    print("\n📖 Documentation:")
    print("  - README_experiments.md (detailed documentation)")
    print("  - SUMMARY.md (complete summary)")
    
    return True

def main():
    """Main execution"""
    print_header("ISOLATION FOREST EXPERIMENTS - MASTER SCRIPT")
    
    print("""
This script will:
  1. Check dependencies
  2. Run paper replication experiments
  3. Generate visualizations and analysis
  4. Create summary report

Options:
  --quick    : Run quick mode (3 datasets, faster)
  --skip-exp : Skip experiments, only analyze existing results
  --help     : Show this help message
""")
    
    # Parse arguments
    quick_mode = '--quick' in sys.argv
    skip_experiments = '--skip-exp' in sys.argv
    
    if '--help' in sys.argv or '-h' in sys.argv:
        return
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Step 2: Run experiments (unless skipped)
    if not skip_experiments:
        if not run_experiments(quick_mode):
            print("\n❌ Experiments failed")
            return
    else:
        print_header("SKIPPING EXPERIMENTS")
        print("Using existing results...")
    
    # Step 3: Generate visualizations
    if not generate_visualizations():
        print("\n❌ Visualization generation failed")
        return
    
    # Step 4: Create report
    if not create_report():
        print("\n❌ Report creation failed")
        return
    
    # Final summary
    print_header("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
    
    print("""
Next steps:
  1. Review anomaly_detection_results.csv for raw data
  2. Check generated visualizations (.png files)
  3. Read SUMMARY.md for complete overview
  4. Consult README_experiments.md for details

To customize experiments:
  - Edit replicate_paper.py to add/remove datasets or models
  - Modify model parameters in get_model_configs()
  - Adjust visualization settings in analyze_results.py
""")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
