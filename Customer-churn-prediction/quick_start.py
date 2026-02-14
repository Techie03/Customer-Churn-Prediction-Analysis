"""
Quick Start Guide - Customer Churn Prediction
==============================================
Run this script to execute the complete analysis pipeline
"""

import subprocess
import sys

print("=" * 70)
print("CUSTOMER CHURN PREDICTION - QUICK START")
print("=" * 70)

def install_requirements():
    """Install required packages"""
    print("\n[1] Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                             "pandas", "numpy", "scikit-learn", "matplotlib", 
                             "seaborn", "imbalanced-learn"])
        print("✓ All requirements installed successfully")
        return True
    except Exception as e:
        print(f"✗ Error installing requirements: {e}")
        return False

def run_main_analysis():
    """Execute main churn prediction analysis"""
    print("\n[2] Running main analysis...")
    try:
        subprocess.check_call([sys.executable, "customer_churn_prediction.py"])
        print("✓ Main analysis completed successfully")
        return True
    except Exception as e:
        print(f"✗ Error in main analysis: {e}")
        return False

def generate_visualizations():
    """Generate visualization plots"""
    print("\n[3] Generating visualizations...")
    try:
        subprocess.check_call([sys.executable, "visualizations.py"])
        print("✓ Visualizations generated successfully")
        return True
    except Exception as e:
        print(f"✗ Error generating visualizations: {e}")
        return False

def main():
    """Run complete pipeline"""
    print("\nStarting Customer Churn Prediction Pipeline...\n")
    
    # Step 1: Install requirements
    if not install_requirements():
        print("\n⚠ Failed to install requirements. Please install manually.")
        return
    
    # Step 2: Run analysis
    if not run_main_analysis():
        print("\n⚠ Failed to run main analysis.")
        return
    
    # Step 3: Generate visualizations
    if not generate_visualizations():
        print("\n⚠ Failed to generate visualizations.")
        return
    
    # Success message
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  • model_comparison.csv - Model performance metrics")
    print("  • feature_importance.csv - Feature rankings")
    print("  • predictions.csv - Test predictions with probabilities")
    print("  • churn_analysis_visualizations.png - Main dashboard")
    print("  • detailed_analysis_plots.png - Additional insights")
    print("\nNext Steps:")
    print("  1. Review the CSV files for detailed metrics")
    print("  2. Open PNG files to view visualizations")
    print("  3. Examine feature_importance.csv for key drivers")
    print("  4. Use predictions.csv for model deployment planning")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
