"""
Customer Churn Prediction - Visualizations
===========================================
Generate comprehensive visualizations for churn analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

print("Generating visualizations for Customer Churn Analysis...")

# ============================================================================
# Load saved results
# ============================================================================

try:
    model_comparison = pd.read_csv('/home/claude/model_comparison.csv')
    feature_importance = pd.read_csv('/home/claude/feature_importance.csv')
    predictions = pd.read_csv('/home/claude/predictions.csv')
    print("✓ Data loaded successfully")
except:
    print("⚠ Run customer_churn_prediction.py first to generate data")
    exit()

# ============================================================================
# Create visualizations
# ============================================================================

fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison - Accuracy
ax1 = plt.subplot(2, 3, 1)
colors = ['#FF6B6B' if acc < 0.84 else '#4ECDC4' for acc in model_comparison['Accuracy']]
bars = ax1.barh(model_comparison['Model'], model_comparison['Accuracy'], color=colors)
ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xlim([0.70, 0.90])
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax1.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{width:.3f}', ha='left', va='center', fontweight='bold')
ax1.axvline(x=0.84, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Target: 84%')
ax1.legend()

# 2. All Metrics Comparison
ax2 = plt.subplot(2, 3, 2)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(model_comparison))
width = 0.15

for i, metric in enumerate(metrics):
    ax2.bar(x + i*width, model_comparison[metric], width, label=metric, alpha=0.8)

ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_title('All Metrics Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x + width * 2)
ax2.set_xticklabels(model_comparison['Model'], rotation=45, ha='right')
ax2.legend(loc='lower right')
ax2.set_ylim([0.70, 1.0])
ax2.grid(axis='y', alpha=0.3)

# 3. Top 10 Feature Importance
ax3 = plt.subplot(2, 3, 3)
top_features = feature_importance.head(10)
colors_feat = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
bars = ax3.barh(top_features['feature'], top_features['importance'], color=colors_feat)
ax3.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax3.set_title('Top 10 Most Important Features', fontsize=14, fontweight='bold')
ax3.invert_yaxis()
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax3.text(width + 0.002, bar.get_y() + bar.get_height()/2, 
            f'{width:.4f}', ha='left', va='center', fontsize=9)

# 4. Confusion Matrix Heatmap
ax4 = plt.subplot(2, 3, 4)
cm = confusion_matrix(predictions['actual'], predictions['predicted'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
           xticklabels=['No Churn', 'Churn'],
           yticklabels=['No Churn', 'Churn'], ax=ax4,
           annot_kws={'size': 14, 'weight': 'bold'})
ax4.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax4.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax4.set_title('Confusion Matrix - Best Model', fontsize=14, fontweight='bold')

# Add accuracy and metrics to confusion matrix
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
ax4.text(0.5, -0.15, f'Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%}',
        ha='center', transform=ax4.transAxes, fontsize=10, fontweight='bold')

# 5. ROC Curve
ax5 = plt.subplot(2, 3, 5)
fpr, tpr, _ = roc_curve(predictions['actual'], predictions['churn_probability'])
roc_auc = auc(fpr, tpr)
ax5.plot(fpr, tpr, color='#4ECDC4', lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
ax5.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
ax5.set_xlim([0.0, 1.0])
ax5.set_ylim([0.0, 1.05])
ax5.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax5.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax5.set_title('ROC Curve - Random Forest (Tuned)', fontsize=14, fontweight='bold')
ax5.legend(loc="lower right", fontsize=10)
ax5.grid(alpha=0.3)

# 6. Prediction Distribution
ax6 = plt.subplot(2, 3, 6)
churn_probs = predictions[predictions['actual'] == 1]['churn_probability']
no_churn_probs = predictions[predictions['actual'] == 0]['churn_probability']
ax6.hist(no_churn_probs, bins=30, alpha=0.6, label='No Churn (Actual)', color='#4ECDC4', edgecolor='black')
ax6.hist(churn_probs, bins=30, alpha=0.6, label='Churn (Actual)', color='#FF6B6B', edgecolor='black')
ax6.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Decision Threshold')
ax6.set_xlabel('Predicted Churn Probability', fontsize=12, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax6.set_title('Churn Probability Distribution', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/churn_analysis_visualizations.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved to: churn_analysis_visualizations.png")

# ============================================================================
# Create additional analysis plots
# ============================================================================

fig2, axes = plt.subplots(2, 2, figsize=(16, 12))

# Feature importance cumulative
ax1 = axes[0, 0]
cumsum = feature_importance.head(15)['importance'].cumsum()
ax1.plot(range(1, len(cumsum)+1), cumsum, marker='o', linewidth=2, markersize=8, color='#4ECDC4')
ax1.fill_between(range(1, len(cumsum)+1), cumsum, alpha=0.3, color='#4ECDC4')
ax1.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cumulative Importance', fontsize=12, fontweight='bold')
ax1.set_title('Cumulative Feature Importance (Top 15)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0.8, color='red', linestyle='--', label='80% threshold')
ax1.legend()

# Model metrics radar chart
ax2 = axes[0, 1]
best_model = model_comparison.iloc[-1]  # Assuming last row is tuned RF
categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
values = [best_model[cat] for cat in categories]
values += values[:1]  # Complete the circle

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

ax2 = plt.subplot(2, 2, 2, projection='polar')
ax2.plot(angles, values, 'o-', linewidth=2, color='#4ECDC4', label='Random Forest (Tuned)')
ax2.fill(angles, values, alpha=0.25, color='#4ECDC4')
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, fontsize=10)
ax2.set_ylim(0.7, 1.0)
ax2.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax2.grid(True)

# Precision-Recall curve
ax3 = axes[1, 0]
from sklearn.metrics import precision_recall_curve
precision_vals, recall_vals, _ = precision_recall_curve(predictions['actual'], 
                                                         predictions['churn_probability'])
ax3.plot(recall_vals, precision_vals, linewidth=3, color='#FF6B6B')
ax3.fill_between(recall_vals, precision_vals, alpha=0.3, color='#FF6B6B')
ax3.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax3.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])

# Model comparison spider
ax4 = axes[1, 1]
x = np.arange(len(model_comparison))
width = 0.35
accuracy_vals = model_comparison['Accuracy'] * 100
f1_vals = model_comparison['F1-Score'] * 100

bars1 = ax4.bar(x - width/2, accuracy_vals, width, label='Accuracy %', 
               color='#4ECDC4', alpha=0.8, edgecolor='black')
bars2 = ax4.bar(x + width/2, f1_vals, width, label='F1-Score %', 
               color='#FF6B6B', alpha=0.8, edgecolor='black')

ax4.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax4.set_title('Accuracy vs F1-Score by Model', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(model_comparison['Model'], rotation=45, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim([70, 90])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/detailed_analysis_plots.png', dpi=300, bbox_inches='tight')
print("✓ Detailed analysis plots saved to: detailed_analysis_plots.png")

print("\n" + "="*70)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*70)
print("\nGenerated files:")
print("  1. churn_analysis_visualizations.png - Main dashboard")
print("  2. detailed_analysis_plots.png - Additional insights")
