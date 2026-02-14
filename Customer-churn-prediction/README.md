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
```bash
python customer_churn_prediction.py
```

2. **Generate visualizations:**
```bash
python visualizations.py
```

## 📋 Dataset Features (20+ Variables)

### Demographics
- Age
- Gender
- Senior Citizen status

### Account Information
- Tenure (months)
- Contract Type (Month-to-Month, One Year, Two Year)
- Payment Method
- Paperless Billing

### Services
- Phone Service
- Multiple Lines
- Internet Service (DSL, Fiber Optic)
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV & Movies

### Financial
- Monthly Charges
- Total Charges

### Engagement Metrics
- Customer Service Calls
- Late Payments
- Number of Services

### Engineered Features
- Tenure Category
- Charges to Tenure Ratio
- Service Usage Score
- Customer Loyalty Score
- High Risk Indicator
- Payment Reliability Score

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing
- Handled missing values
- Encoded categorical variables using Label Encoding
- Created 5+ engineered features
- Applied StandardScaler for feature scaling

### 2. Train-Test Split
- 80% training, 20% testing
- Stratified split to maintain class distribution

### 3. Class Imbalance Handling
- **Before SMOTE:** 73% No Churn, 27% Churn
- **After SMOTE:** Balanced 50-50 split
- **Impact:** 22% improvement in minority class recall

### 4. Model Training

#### Baseline Models:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier** (100 estimators)

### 5. Hyperparameter Tuning

Used GridSearchCV on Random Forest with:
- **n_estimators:** [100, 200, 300]
- **max_depth:** [10, 20, 30, None]
- **min_samples_split:** [2, 5, 10]
- **min_samples_leaf:** [1, 2, 4]
- **max_features:** ['sqrt', 'log2']

**Cross-validation:** 5-fold CV
**Scoring metric:** Accuracy

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | ~78% | ~0.76 | ~0.72 | ~0.74 | ~0.85 |
| Decision Tree | ~80% | ~0.78 | ~0.75 | ~0.76 | ~0.83 |
| Random Forest | ~82% | ~0.81 | ~0.78 | ~0.79 | ~0.88 |
| **Random Forest (Tuned)** | **~84%** | **~0.83** | **~0.81** | **~0.82** | **~0.90** |

## 🎯 Key Findings

### Top Risk Factors for Churn:
1. Contract Type (Month-to-Month highest risk)
2. Tenure (< 12 months)
3. Customer Service Calls (> 3)
4. Payment Method (Electronic Check)
5. Monthly Charges (> $80)
6. Lack of Tech Support
7. Late Payments

### Model Insights:
- Random Forest outperformed all baseline models
- SMOTE significantly improved recall for churn class
- Feature engineering contributed to better model performance
- GridSearchCV found optimal hyperparameters improving accuracy by 2%

## 📊 Visualizations

The project generates comprehensive visualizations including:

1. **Model Comparison Charts** - Accuracy across all models
2. **Feature Importance** - Top 10 predictive features
3. **Confusion Matrix** - Classification performance breakdown
4. **ROC Curve** - Model discrimination capability
5. **Prediction Distribution** - Probability distributions by class
6. **Precision-Recall Curve** - Trade-off analysis

## 💼 Business Applications

### Proactive Retention Strategy
- Identify high-risk customers before they churn
- Target intervention efforts efficiently
- Reduce customer acquisition costs

### Risk Segmentation
- Categorize customers by churn probability
- Customize retention offers
- Prioritize customer success resources

### Feature Monitoring
- Track key indicators (tenure, service calls, payment behavior)
- Set up early warning alerts
- Implement preventive measures

## 🔄 Model Deployment Recommendations

1. **Real-time Scoring:** Deploy model as API endpoint
2. **Batch Processing:** Weekly churn risk assessments
3. **Monitoring:** Track model performance metrics
4. **Retraining:** Quarterly model updates with new data
5. **A/B Testing:** Compare intervention strategies

## 📝 Code Highlights

### SMOTE Implementation
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
```

### GridSearchCV for Hyperparameter Tuning
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
```

### Feature Engineering Example
```python
df['loyalty_score'] = (df['tenure_months'] * 0.5) - (df['customer_service_calls'] * 2)
df['high_risk'] = ((df['contract_type'] == 'Month-to-Month') & 
                   (df['tenure_months'] < 12)).astype(int)
```

## 📚 Model Evaluation Metrics

- **Accuracy:** Overall correctness of predictions
- **Precision:** Proportion of positive predictions that are correct
- **Recall:** Proportion of actual positives correctly identified
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Model's ability to distinguish between classes

## Results

<img width="5966" height="3581" alt="churn_analysis_visualizations" src="https://github.com/user-attachments/assets/035fcf72-2e9c-4441-aa59-6c2093fb92fd" />
<img width="4655" height="2120" alt="detailed_analysis_plots" src="https://github.com/user-attachments/assets/25c035a2-972b-40c6-ab38-dea72b9a4245" />



## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end machine learning pipeline development
- Handling imbalanced datasets with SMOTE
- Hyperparameter optimization with GridSearchCV
- Feature engineering and domain knowledge application
- Model comparison and selection
- Business-focused model evaluation

## 🤝 Contributing

Suggestions for improvements:
- Additional ensemble methods (XGBoost, LightGBM)
- Deep learning approaches
- Time-series analysis for temporal patterns
- Customer segmentation clustering
- Explainable AI techniques (SHAP, LIME)

## 📞 Contact

For questions or feedback about this project, please reach out through GitHub issues.

## 📄 License

This project is open source and available for educational purposes.

---

**Note:** This analysis uses synthetic data generated to match real-world churn patterns. For production use, replace with actual customer data while ensuring proper data privacy and compliance.
