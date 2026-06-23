import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0b1220; color: #e6e6e6; }
    .verdict-box {
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .fraud { background-color: #3a1414; border: 1px solid #b33; color: #ff8080; }
    .legit { background-color: #133a1a; border: 1px solid #2d7a3a; color: #7fdb8f; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("Models/fraud_rf_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Data/demo_transactions.csv")

model = load_model()
data = load_data()

st.title("Credit Card Fraud Detection")
st.caption(
    "V1-V28 are PCA-anonymized features provided by the dataset (Kaggle, ULB). "
    "Since these can't be meaningfully hand-crafted, this demo runs the model "
    "on real transactions pulled from the test set instead of manual input."
)

feature_cols = [c for c in data.columns if c != "true_label"]

options = [f"Transaction {i+1}" for i in range(len(data))]
choice = st.selectbox("Pick a transaction", options)
row_idx = options.index(choice)
row = data.iloc[row_idx]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Feature values")
    st.dataframe(row[feature_cols].to_frame(name="value"), use_container_width=True)

with col2:
    st.subheader("Prediction")
    X_row = row[feature_cols].values.reshape(1, -1)
    pred = model.predict(X_row)[0]
    prob = model.predict_proba(X_row)[0][1]

    verdict = "FRAUD" if pred == 1 else "LEGIT"
    css_class = "fraud" if pred == 1 else "legit"

    st.markdown(
        f'<div class="verdict-box {css_class}">Model verdict: {verdict}<br>Fraud probability: {prob:.4f}</div>',
        unsafe_allow_html=True
    )

    actual = "FRAUD" if row["true_label"] == 1 else "LEGIT"
    st.write(f"Actual label: **{actual}**")
    st.write("Correct" if pred == row["true_label"] else "Misclassified")

st.divider()
st.caption("Random Forest, trained on SMOTE-balanced training data, evaluated on the original imbalanced test set.")