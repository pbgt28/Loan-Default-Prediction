import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Loan Default Prediction", layout="wide")

@st.cache_resource
def load_pipeline():
    return joblib.load("loan_default_pipeline.joblib")

pipeline = load_pipeline()

st.title("🏦 Loan Default Risk Assessment")
st.markdown("Enter applicant financial and demographic details to assess the risk of loan default.")

# Form layout matching the 16 features
with st.form("loan_application_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        education = st.selectbox("Education", ["Bachelor's", "High School", "Master's", "PhD"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

    with col2:
        st.subheader("Employment & Income")
        income = st.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=75000, step=1000)
        employment_type = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
        months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=48)
        mortgage = st.selectbox("Has Mortgage?", ["No", "Yes"])
        co_signer = st.selectbox("Has Co-Signer?", ["No", "Yes"])

    with col3:
        st.subheader("Loan & Credit Details")
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=500000, value=50000, step=1000)
        loan_term = st.selectbox("Loan Term (Years)", [1, 2, 3, 4, 5], index=2)
        interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=12.5, step=0.1)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
        num_credit_lines = st.selectbox("Number of Credit Lines", [1, 2, 3, 4], index=1)
        loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])

    submitted = st.form_submit_button("Predict Default Risk")

if submitted:
    # Build payload with exact column names as used during training
    input_df = pd.DataFrame([{
        'Age': age,
        'Income': income,
        'Loan_Amount': loan_amount,
        'Credit_Score': credit_score,
        'Num_of_Months_Employed': months_employed,
        'Num_of_Credit_Lines': num_credit_lines,
        'Interest_Rate': interest_rate,
        'Loan_Term': loan_term,
        'DTI_Ratio': dti_ratio,
        'Education': education,
        'Employment_Type': employment_type,
        'Marital_Status': marital_status,
        'Mortgage': mortgage,
        'Dependents': dependents,
        'Loan_Purpose': loan_purpose,
        'Co_Signer': co_signer
    }])

    # Predict probability of default (Class 1)
    proba = pipeline.predict_proba(input_df)[0][1]
    
    # Cost-sensitive threshold (e.g., 0.35 due to the 11.6% default imbalance)
    threshold = 0.35
    is_default = proba >= threshold

    st.divider()
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(label="Default Probability", value=f"{proba * 100:.2f}%")
        
    with res_col2:
        if is_default:
            st.error("⚠️ **High Risk Applicant** — Application Flagged for Rejection / Manual Review.")
        else:
            st.success("✅ **Low Risk Applicant** — Eligible for Approval.")
