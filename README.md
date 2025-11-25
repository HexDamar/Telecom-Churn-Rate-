# Telecom Churn Rate

## 🌟 Overview
This repository contains an end-to-end machine learning pipeline for predicting customer churn in telecommunications companies. The project implements a complete ML workflow from data exploration to deployment, providing actionable insights to improve customer retention strategies in the competitive telecom market.

Live Demo
Experience the application here: [App](https://huggingface.co/spaces/HeXzoE/TELECOM_churn_detection)
---

## 🚀 Problem Statement
**"How can telecommunications companies leverage machine learning to predict customer churn and implement proactive retention strategies?"**

### Market Context:
- High customer acquisition costs in telecom industry (5x retention cost)
- 26.5% churn rate represents significant revenue loss
- Need for data-driven customer retention strategies

### Objectives:
1. Predict customer churn probability with high accuracy 
2. Identify key factors driving customer churn 
3. Provide actionable insights for retention campaigns 
4. Deploy real-time prediction system for business use 

---

## 📂 Repository Structure
```
customer-churn-prediction/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── src/
    ├── __init__.py                 # Package initialization
    ├── prediction.py               # Prediction interface and logic
    ├── eda.py                      # Exploratory Data Analysis dashboard
    └── model.pkl                   # Trained Gradient Boosting model
├── assets/
    └── images/                     # EDA visualizations
        ├── pic 1.png              # Churn Distribution
        ├── pic 2.png              # Gender Distribution
        ├── pic 3.png              # Contract Analysis
        ├── pic 4.png              # Payment Method Distribution
        ├── pic 5.png              # Payment Method vs Churn
        └── pic 6.png              # Monthly Charges Analysis
```


---

## 🔍 Key Insights
### 📈 Churn Analysis
- **Overall Churn Rate**: 26.5% (1,869 customers) 📉
- **Retention Rate**: 73.5% (5,174 customers) ✅
- **Critical Segments**: Month-to-month contracts & Electronic check users 🚨

### 📊 Customer Behavior Patterns
1. **Payment Method Impact**: Electronic check users have 45.3% churn rate 💳
2. **Contract Sensitivity**: Month-to-month contracts show 42.7% churn rate 📝
3. **Tenure Correlation**: New customers (<12 months) highest churn risk ⏳

---

## 🛠️ Methodology
### 🔧 Machine Learning Pipeline Architecture
1. Data Processing:
Handle missing values in TotalCharges (structural zeros)
Feature engineering for categorical and numerical variables
Outlier detection using IQR and standard deviation

2. Model Training & Selection:
- Tested 6 algorithms: KNN, SVC, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- Feature selection using ANOVA F-test and Chi-square tests
- Hyperparameter tuning for optimal performance

3. Model Deployment:
- Real-time prediction interface using Gradio
- EDA dashboard for business insights
- Production-ready pipeline

## 📊 Performance Metrics
- Best Model: Gradient Boosting Classifier 🏆
- AUC Score: 0.865 (Excellent discrimination)
- F1 Score: 0.743 (Test Set)
- Cross-validation: 0.7257 ± 0.0163

---

## 💡 Business Recommendations
1. Retention Strategy
Target month-to-month contract customers with upgrade incentives
Implement proactive outreach for high-risk segments

2. Payment Optimization
Migrate electronic check users to automatic payment methods
Offer incentives for auto-pay enrollment

3. Customer Segmentation
Focus retention efforts on customers with tenure <12 months
Develop personalized offers based on usage patterns

---

## ⚙️ Tech Stack
| Category          | Tools/Libraries |
|-------------------|-----------------|
| **Machine Learning**          | Scikit-learn, XGBoost, Pandas, NumPy |
| **Web Framework**     | Gradio |
| **Deployement**   | Hugging Face Spaces |
| **Visualization**| Matplotlib, Seaborn |

```python
# Core Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import gradio as gr
```

## 📊 Business Impact
1. Revenue Protection: Prevent loss from 1,869 churning customers
2. Cost Efficiency: Retention cost 5x cheaper than acquisition
3. Customer Insights: Data-driven understanding of churn drivers
4. Competitive Advantage: Proactive customer retention capability
---

## ✨ Contributors
[Nugroho Wicaksono](https://github.com/HexDamar) - Data Scientist & Machine Learning Engineer  

🔹 *Last Updated: November 2025*
