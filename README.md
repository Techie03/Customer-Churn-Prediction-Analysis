# Customer Churn Prediction Analysis

## Project Overview
A comprehensive machine learning project that predicts customer churn using classification models including Logistic Regression, Decision Trees, and Random Forest. The project achieves **84% accuracy** through hyperparameter tuning and handles class imbalance using SMOTE technique.

## 📊 Key Achievements

- ✅ **84% Accuracy** with Random Forest after GridSearchCV hyperparameter tuning
- ✅ **7,000+ subscriber records** processed with 20+ features
- ✅ **SMOTE implementation** to address class imbalance (73% vs 27%)
- ✅ **22% improvement** in minority class recall through balanced sampling
- ✅ **Comprehensive feature engineering** creating 5+ derived features

## 🛠️ Technologies Used

- **Python 3.8+**
- **Scikit-learn** - Machine learning models and evaluation
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **imbalanced-learn** - SMOTE for handling class imbalance

## 📁 Project Structure

```
customer-churn-prediction/
│
├── customer_churn_prediction.py    # Main analysis script
├── visualizations.py                # Visualization generation
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
│
├── Output Files:
│   ├── model_comparison.csv         # Model performance metrics
│   ├── feature_importance.csv       # Feature importance rankings
│   ├── predictions.csv              # Test set predictions
│   ├── churn_analysis_visualizations.png
│   └── detailed_analysis_plots.png
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Running the Analysis

1. **Run the main analysis:**
