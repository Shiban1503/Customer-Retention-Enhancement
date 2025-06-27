# app.py - Updated for custom model filenames
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Set up the app
st.set_page_config(
    page_title="SmartBank Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# Load model and preprocessing artifacts
@st.cache_resource
def load_model():
    model = joblib.load('optimized_churn_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    return model, preprocessor

model, preprocessor = load_model()

# UI
st.title("🏦 SmartBank Customer Churn Prediction Dashboard")
st.markdown("""
Predict which customers are at risk of churning and take proactive retention measures.
""")

# Create input form
with st.form("customer_form"):
    st.subheader("🧾 Customer Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 80, 45)
        income = st.selectbox("Income Level", ["Low", "Medium", "High"])
        total_spent = st.number_input("Total Spent (£)", 
                                    min_value=0.0, value=500.0)
        gender = st.radio("Gender", ["Male", "Female"])
        marital = st.selectbox("Marital Status", 
                                    ["Single", "Married", "Divorced", "Widowed"])
        service_usage = st.selectbox("Primary Service Usage",
                                   ["Online Banking", "Mobile App", "Branch"])
        
    with col2:
        login_freq = st.slider("Login Frequency (per month)", 1, 50, 15)

        num_transactions = st.slider("Number of Transactions", 
                                   1, 50, 10)
        resolution_rate = st.slider("Service Resolution Rate", 0.0, 1.0, 0.85)
        total_interactions = st.slider("Total Service Interactions", 0, 20, 3)
        unique_cats = st.slider("Unique Categories Purchased", 1, 10, 3)
    
    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    # Build input df
    input_df = pd.DataFrame([{
        'Age': age,
        'Gender': gender,
        'MaritalStatus': marital,
        'ServiceUsage': service_usage,
        'IncomeLevel': income,
        'LoginFrequency': login_freq,
        'TotalSpent': total_spent,
        'TotalInteractions': total_interactions,
        'ResolutionRate': resolution_rate,
        'NumTransactions': num_transactions,
        'UniqueCategories': unique_cats,
        'AvgSpent': total_spent / (1 if login_freq == 0 else login_freq),
        'ResolvedInteractions': int(total_interactions * resolution_rate),
        'ValuePerLogin': total_spent / (1 + login_freq),
        'ResolutionDeficit': 1 - resolution_rate,
        'SpendPerTransaction':total_spent / num_transactions
    }])
 
    # Preprocess
    X_proc = preprocessor.transform(input_df)

    # Predict
    prob = model.predict_proba(X_proc)[0][1]
    prediction = model.predict(X_proc)[0]

    # Display results
    st.subheader("📊 Prediction Result")

    # Create metrics columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Churn Probability", f"{prob:.1%}")
    
    with col2:
        status = "At Risk 🚨" if prediction == 1 else "Retained ✅"
        st.metric("Predicted Status", status)
    
    with col3:
        risk_level = "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low")
        st.metric("Risk Level", risk_level)

    # Explanation section
    st.subheader("Model Explanation")

    # Explain prediction
    classifier = model.named_steps["classifier"]
    st.subheader("🔍 Top Feature Influences")

    model_type = type(classifier).__name__
    try:
        if model_type in ["RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier"]:
            explainer = shap.Explainer(classifier)
            shap_values = explainer(X_proc)
            st.set_option("deprecation.showPyplotGlobalUse", False)
            shap.summary_plot(shap_values, X_proc, feature_names=preprocessor.get_feature_names_out(), plot_type="bar")
            st.pyplot(bbox_inches="tight")

        elif model_type == "LogisticRegression":
            coef = classifier.coef_[0]
            features = preprocessor.get_feature_names_out()
            clean_features = [f.split("__")[-1] for f in features]
            top_features = pd.Series(coef * X_proc[0], index=clean_features).sort_values(key=abs, ascending=False).head(5)
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.barplot(x=top_features.values, y=top_features.index, palette="coolwarm", ax=ax)
            ax.set_title("Top Contributing Features")
            ax.set_xlabel("Feature Impact (Weighted Value)")
            ax.set_ylabel("Feature Name")
            st.pyplot(fig)
        else:
            st.info("Model explanation not supported for this model type.")
    except Exception as e:
        st.warning(f"Model explanation failed: {str(e)}")

    # Recommendations
    st.subheader("📌 Recommended Retention Actions")
    
    if prediction == 1:
        if prob > 0.7:
            st.warning("**Immediate action required** - High-value customer at significant risk")
            st.write("""
            - Assign dedicated relationship manager
            - Offer personalized retention package
            - Schedule executive callback within 24 hours
            """)
        else:
            st.warning("**Proactive engagement recommended**")
            st.write("""
            - Send customer satisfaction survey
            - Offer product consultation
            - Provide loyalty program benefits
            """)
        
        if input_df['ResolutionRate'].values[0] < 0.7:
            st.info("**Service Improvement Opportunity**")
            st.write(f"""
            - Customer has low resolution rate ({input_df['ResolutionRate'].values[0]:.0%})
            - Prioritize quick resolution for any open tickets
            - Offer service recovery compensation
            """)
    else:
        st.success("**Customer appears satisfied** - Maintain regular engagement")
        st.write("""
        - Continue monitoring engagement metrics
        - Include in standard marketing campaigns
        - Consider cross-sell opportunities
        """)
    
    # Save to database (simulated)
    if st.button("Save Prediction to CRM"):
        st.success("Prediction saved to customer record!")

# How to use section
with st.expander("ℹ️ How to use this dashboard"):
    st.markdown("""
    1. **Enter customer details** in the form
    2. Click **Predict Churn Risk** to see the prediction
    3. Review the **risk level** and **key factors**
    4. Take **recommended actions** to retain at-risk customers
    5. Save results to your CRM system (simulated)
    
    """)

# Add footer
st.markdown("---")
st.caption("SmartBank Customer Retention System | v1.0")