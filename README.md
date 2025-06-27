# Customer Retention Enhancement through Predictive Analytics 

## 📚 Table of Contents
- [Purpose](#purpose)
- [Dataset Overview](#dataset-overview)
- [Steps and Implementation](#steps-and-implementation)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Machine Learning - Predictive Model](#machine-learning---predictive-model)
  - [Training and Evaluation](#training-and-evaluation)
  - [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)
- [Conclusion](#conclusion)

---

## 🎯 Purpose
The project addresses the challenge of customer churn at SmartBank. Using historical customer data, we aim to:
- Predict churn likelihood using machine learning
- Explain factors contributing to churn
- Recommend customer retention strategies

---

## 📊 Dataset Overview
The dataset consists of anonymised customer records including:
- Demographics (Age, Gender, Marital Status, Income Level)
- Usage patterns (Login frequency, Channel preferences)
- Transaction behavior (Spending, Category diversity)
- Customer service interaction history
- Churn status labels (target variable)

Raw data file: `data/Customer_Churn_Data_Large.xlsx`

---

## 🧪 Steps and Implementation

### 🔄 Data Preprocessing
- Merged five separate tables into a unified customer-level view
- Handled missing values (imputation and flagging)
- Detected and capped outliers using IQR
- Scaled numerical features and encoded categoricals
- Saved cleaned train/test splits to CSV

### 📊 Exploratory Data Analysis (EDA)
- Plotted churn distribution across income, age, gender
- Identified churn correlations with service use and unresolved support tickets
- Visualised boxplots, KDE plots, and heatmaps for relationships

### 🤖 Machine Learning - Predictive Model
- Binary classification task (Churn vs No Churn)
- Models evaluated: Logistic Regression, Random Forest, XGBoost
- Used SMOTE for class imbalance handling
- Pipeline included hyperparameter tuning via GridSearchCV

### 🎯 Training and Evaluation
- Metrics: Precision, Recall, F1-score, ROC-AUC, PR-AUC
- Best model: XGBoost with ~0.88 ROC-AUC and strong recall
- Feature importance analysed using SHAP and `feature_importances_`

### 🧾 Results
- Deployed a Streamlit app for business use
- Real-time churn prediction with explanations
- Business recommendations generated dynamically based on top features

---

## 💻 Installation
```bash
pip install -r requirements.txt
```

## ▶️ Usage
```bash
cd app
streamlit run app.py
```

Upload a CSV matching the structure of `test_churn_data.csv` and receive churn predictions with explanations.

---

## 🧱 Project Structure
```bash
├── app/                       # Streamlit app and assets
│   ├── app.py
│   ├── optimized_churn_model.pkl
│   ├── preprocessor.pkl
│   ├── test_churn_data.csv
│   └── business_recommendations.json
│
├── notebooks/                # Model development and EDA
│   └── Customer_Retention_Enhancement.ipynb
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

## 📄 License
[MIT](LICENSE)

---

## ✅ Conclusion
This project demonstrates how predictive analytics and interpretability tools can be combined to drive customer-centric decision-making. It empowers banks to proactively intervene and reduce churn risk, supported by transparent, data-driven insights.

Developed by **Mohamed Shiban Lal**
_Data Science Graduate | Predictive Analytics Enthusiast_
