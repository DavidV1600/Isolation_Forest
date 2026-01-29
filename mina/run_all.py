#!/usr/bin/env python3
"""
Master script to run all isolation forest experiments

This script orchestrates the complete experimental pipeline:
1. Runs paper replication experiments (via replicate_paper.py)
2. Generates visualizations and analysis (via analyze_results.py)
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

def main():
    """Main execution"""
    print_header("ISOLATION FOREST EXPERIMENTS - MASTER SCRIPT")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first")
        return
    
    # Step 2: Run experiments
    print_header("RUNNING EXPERIMENTS (REPLICATION)")
    start_time = datetime.now()
    
    if not run_command("python replicate_paper.py", "Paper replication experiments"):
        print("\n❌ Experiments failed")
        return
        
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n⏱ Experiments completed in {duration:.1f} seconds")
    
    # Step 3: Generate visualizations
    print_header("GENERATING ANALYSIS & VISUALIZATION")
    if not run_command("python analyze_results.py", "Analysis and visualization"):
        print("\n❌ Visualization generation failed")
        return
    
    # Final summary
    print_header("🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
    print("""
Outputs:
  - anomaly_detection_results.csv (raw data)
  - comparison_roc_auc.png
  - comparison_pr_auc.png
  - heatmap_roc_auc.png
  - time_vs_performance.png
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