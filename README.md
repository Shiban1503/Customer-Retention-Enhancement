# 🧠 Customer Churn Prediction for SmartBank (Lloyds Banking Group)

A predictive analytics project focused on identifying customers at risk of churning using machine learning, empowering SmartBank to implement proactive retention strategies.

## 📑 Table of Contents
- [Purpose](#purpose)
- [Dataset Overview](#dataset-overview)
- [Steps and Implementation](#steps-and-implementation)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Machine Learning - Predictive Model](#machine-learning---predictive-model)
  - [Training and Evaluation](#training-and-evaluation)
- [📊 Results](#results)
- [⚙️ Installation](#installation)
- [🚀 Usage](#usage)
- [📁 Project Structure](#project-structure)
- [📄 License](#license)
- [✅ Conclusion](#conclusion)

---

## 🎯 Purpose

This project aims to help SmartBank (a subsidiary of Lloyds Banking Group) improve customer retention by identifying customers likely to churn. Using machine learning, we developed an end-to-end data science pipeline to provide actionable insights and support strategic decision-making.

---

## 📊 Dataset Overview

The project uses a synthetic but realistic dataset containing:
- Demographics (Age, Gender, Marital Status, Income Level)
- Usage patterns (Login frequency, Channel preferences)
- Transaction behavior (Spending, Category diversity)
- Customer service interaction history
- Churn status labels (target variable)

Raw data file: `Customer_Churn_Data_Large.xlsx`

---


## 🧪 Steps and Implementation

### 🔄 Data Preprocessing
- Aggregated multiple sources: demographics, service logs, transactions, churn labels
- Engineered key features: `TotalSpent`, `LoginFrequency`, `ResolutionRate`, `ValuePerLogin`
- Handled missing values and capped outliers using IQR method
- Standardized numeric features and encoded categoricals using pipelines
- Train-test split: stratified to handle class imbalance

### 📊 Exploratory Data Analysis (EDA)
- Churn rate by income, marital status, age, and service usage
- Behavioral trends: churners showed lower spending, fewer logins, lower resolution rates
- Used KDE plots, boxplots, bar charts, and heatmaps

### 🤖 Machine Learning - Predictive Model
- Compared multiple classifiers:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Addressed class imbalance using SMOTE oversampling in pipelines
- Hyperparameter tuning via `GridSearchCV`

### 🎯 Training and Evaluation
- Final model (Best Model): **Logistic Regression with L1 regularization**
- Model metrics on test set:
  - Precision, Recall, F1, ROC-AUC, PR-AUC
- Interpretation using:
  - SHAP for tree models (optional)
  - Coefficients for linear models

### 🧾 Results
The Logistic Regression model was chosen for its:
- High recall on at-risk customers
- Simple and transparent decision logic
- Smooth integration into SmartBank’s workflow

Key features influencing churn included:
- Total spent
- Login frequency
- Service resolution rate

---

## 💻 Installation
```bash
pip install -r requirements.txt
```

## ▶️ Usage
```bash
streamlit run app.py
```
- Enter customer details in the form
- Click Predict Churn Risk to see the prediction
- Review the risk level and key factors
- Take recommended actions to retain at-risk customers
- Save results to your CRM system (simulated)
---

## 🧱 Project Structure
```bash
├── app/                       # Streamlit app and assets
│   ├── app.py
│   ├── optimized_churn_model.pkl
│   ├── preprocessor.pkl
│
├── notebooks/                # Model development and EDA
│   └── Customer Retention Enhancement through Predictive Analytics.ipynb
│
├── data/                     # Raw and processed data
│   ├── Customer_Churn_Data_Large.xlsx
│   └── cleaned_data/
│       ├── train_churn_data.csv
│       └── test_churn_data.csv
│
├── reports/                  # PDF report
│   └── Churn_Analytics_Report.pdf
│
├── scripts/                  # Optional training script
│   └── train_model.py
│
├── requirements.txt
├── LICENSE
└── README.md
```
---

## 📈 Streamlit App Dashboard Output
![Churn Risk Dashboard](https://github.com/Shiban1503/Customer-Retention-Enhancement/blob/main/dashboard_screenshot.png)
![Churn Risk Dashboard](https://github.com/Shiban1503/Customer-Retention-Enhancement/blob/main/dashboard_screenshot_1.png)

---

## 📄 License
This project is licensed under the MIY License. Please refer to the [MIT](LICENSE) file for details.

---

## ✅ Conclusion
This project demonstrates a complete data science pipeline – from data wrangling and EDA to model deployment via Streamlit. It showcases practical skills in churn prediction, model selection, interpretability, and business impact analysis.

---
Developed by **Mohamed Shiban Lal**
_Data Science Graduate | Predictive Analytics Enthusiast_
