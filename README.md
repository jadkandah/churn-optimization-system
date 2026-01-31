# 📊 Customer Churn Prediction & Optimization System

An end-to-end machine learning system for predicting customer churn, explaining model decisions, and optimizing retention strategies under budget constraints.

This project demonstrates the full data science lifecycle, from raw data processing to business decision-making.

---

## 🚀 Features

- Data preprocessing and feature engineering
- Supervised ML model training (Logistic Regression, Random Forest)
- Model evaluation with ROC-AUC, F1-score, Recall
- Explainable AI using SHAP
- Profit-based customer targeting optimization
- Individual customer churn prediction
- Modular, production-style architecture

---

## 📂 Project Structure
```
churn-optimization-system/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── models/
├── reports/
├── src/
│ ├── preprocessing.py
│ ├── train.py
│ ├── explain.py
│ ├── optimize.py
│ └── predict.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 📈 Dataset

Telco Customer Churn Dataset  
Source: IBM / Kaggle

Includes customer demographics, services, billing information, and churn labels.

---

## 🧠 Modeling

- Baseline: Logistic Regression with scaling pipeline
- Alternative: Random Forest
- Train/Test split with stratification
- Evaluation metrics:
  - ROC-AUC
  - F1-score
  - Recall (Churn class)

Final model selected based on recall and business impact.

---

## 🔍 Explainability

Model decisions are explained using SHAP values.

Key churn drivers:

- Customer tenure
- Monthly charges
- Contract type
- Internet service type

SHAP summary plot available in:
- reports/shap_summary.png

---

## 💼 Business Optimization

A profit-based targeting system selects customers for retention offers.

Objective:

Maximize expected profit under budget constraints.

Parameters:

- Discount: $50
- Budget: $10,000
- Retention probability: 40%

Output:
- reports/optimized_targets.csv

---

## 🔮 Prediction

Predict churn probability for individual customers using:
- python src/predict.py

Supports real-time scoring for deployment.

---

## ▶️ Usage

### Install

```bash
pip install -r requirements.txt
```
### Run
```run
python3 src/preprocessing.py
python3 src/train.py
python3 src/explain.py
python3 src/optimize.py
python3 src/predict.py
```
---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- SHAP
- Streamlit
- Joblib

---

## 👤 Author

### Jad Kandah
- LinkedIn: https://www.linkedin.com/in/jad-kandah-992294132
