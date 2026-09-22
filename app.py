import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Default Prediction Dashboard", layout="wide")

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.joblib")
    return joblib.load(model_path)

model = load_model()

st.title("🏦 Loan Default Prediction Dashboard")
st.write("Predict applicant loan default risk using financial and demographic attributes.")

with st.form("loan_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

    with col2:
        st.subheader("Employment & Income")
        income = st.number_input("Annual Income ($)", min_value=1000, max_value=1000000, value=65000, step=1000)
        employment_type = st.selectbox("Employment Type", ["Unemployed", "Self-employed", "Part-time", "Full-time"])
        months_employed = st.number_input("Months Employed", min_value=0, max_value=600, value=48)
        mortgage = st.selectbox("Has Mortgage?", ["No", "Yes"])
        co_signer = st.selectbox("Has Co-Signer?", ["No", "Yes"])

    with col3:
        st.subheader("Loan & Credit Profile")
        loan_amount = st.number_input("Loan Amount ($)", min_value=1000, max_value=500000, value=25000, step=1000)
        loan_term = st.selectbox("Loan Term (Years)", [1, 2, 3, 4, 5], index=2)
        interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=12.5, step=0.1)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        dti_ratio = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
        num_credit_lines = st.selectbox("Number of Credit Lines", [1, 2, 3, 4], index=1)
        loan_purpose = st.selectbox("Loan Purpose", ["Auto", "Business", "Education", "Home", "Other"])

    submitted = st.form_submit_button("Predict Default Risk")

if submitted:
    # 1. Exact numeric mappings used during model training in your notebook
    edu_map = {'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3}
    emp_map = {'Unemployed': 0, 'Self-employed': 1, 'Part-time': 2, 'Full-time': 3}
    marital_map = {'Single': 0, 'Married': 1, 'Divorced': 2}
    purpose_map = {'Auto': 0, 'Business': 1, 'Education': 2, 'Home': 3, 'Other': 4}
    binary_map = {'Yes': 1, 'No': 0}

    # 2. Build DataFrame with exact 16-feature order expected by the model
    features_ordered = [
        'Credit_Score', 'Num_of_Months_Employed', 'Num_of_Credit_Lines', 
        'Interest_Rate', 'Loan_Amount', 'DTI_Ratio', 'Loan_Term', 
        'Marital_Status', 'Education', 'Employment_Type', 
        'Mortgage', 'Dependents', 'Loan_Purpose', 'Co_Signer', 
        'Age', 'Income'
    ]

    input_data = pd.DataFrame([{
        'Credit_Score': credit_score,
        'Num_of_Months_Employed': months_employed,
        'Num_of_Credit_Lines': num_credit_lines,
        'Interest_Rate': interest_rate,
        'Loan_Amount': loan_amount,
        'DTI_Ratio': dti_ratio,
        'Loan_Term': loan_term,
        'Marital_Status': marital_map[marital_status],
        'Education': edu_map[education],
        'Employment_Type': emp_map[employment_type],
        'Mortgage': binary_map[mortgage],
        'Dependents': binary_map[dependents],
        'Loan_Purpose': purpose_map[loan_purpose],
        'Co_Signer': binary_map[co_signer],
        'Age': age,
        'Income': income
    }])[features_ordered]

    # 3. Generate prediction
    proba = model.predict_proba(input_data)[0][1]
    
    # 0.35 threshold accounts for the ~11.6% class imbalance in the training data
    is_default = proba >= 0.35

    st.divider()
    res1, res2 = st.columns(2)
    with res1:
        st.metric("Default Probability", f"{proba * 100:.2f}%")
    with res2:
        if is_default:
            st.error("⚠️ **High Default Risk** — Application flagged for rejection or credit review.")
        else:
            st.success("✅ **Low Default Risk** — Eligible for approval.")
