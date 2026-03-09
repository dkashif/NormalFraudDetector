import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model artifact
artifact = joblib.load("fraud_detection_v2.pkl")
pipeline = artifact["pipeline"]
threshold = artifact["threshold"]

st.set_page_config(page_title="Fraud Detector", page_icon="🔍", layout="centered")
st.title("💳 Credit Card Fraud Detector")
st.markdown(
    """Powered by **XGBoost + SMOTE** trained on 6.3M transactions.  
Enter transaction details below to get a real-time fraud probability."""
)

with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    with col1:
        txn_type = st.selectbox(
            "Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"]
        )
        amount = st.number_input("Amount ($)", min_value=0.0, value=10000.0, step=100.0)
        old_bal_org = st.number_input(
            "Sender Old Balance ($)", min_value=0.0, value=50000.0
        )
        new_bal_org = st.number_input(
            "Sender New Balance ($)", min_value=0.0, value=40000.0
        )
    with col2:
        old_bal_dest = st.number_input(
            "Receiver Old Balance ($)", min_value=0.0, value=0.0
        )
        new_bal_dest = st.number_input(
            "Receiver New Balance ($)", min_value=0.0, value=10000.0
        )
    submitted = st.form_submit_button("🔍 Analyse Transaction")

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "type": txn_type,
                "amount": amount,
                "oldbalanceOrg": old_bal_org,
                "newbalanceOrig": new_bal_org,
                "oldbalanceDest": old_bal_dest,
                "newbalanceDest": new_bal_dest,
                "balanceDiffOrig": old_bal_org - new_bal_org,
                "balanceDiffDest": new_bal_dest - old_bal_dest,
                "origDrained": int(old_bal_org > 0 and new_bal_org == 0),
                "amountEqualsOldBalance": int(amount == old_bal_org),
                "destBalanceMismatch": abs((new_bal_dest - old_bal_dest) - amount),
                "logAmount": np.log1p(amount),
                "destWasEmpty": int(old_bal_dest == 0),
            }
        ]
    )

    prob = pipeline.predict_proba(input_df)[0][1]
    is_fraud = prob >= threshold

    st.divider()
    if is_fraud:
        st.error(f"⚠️ **FRAUD DETECTED** — Confidence: {prob*100:.1f}%")
    else:
        st.success(
            f"✅ **Legitimate Transaction** — Fraud probability: {prob*100:.1f}%"
        )

    st.metric("Fraud Probability", f"{prob*100:.2f}%")
    st.progress(float(prob))
