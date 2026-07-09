import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Healthcare Provider Fraud Detection", layout="wide")

# ---- Load the trained model bundle (model + scaler + threshold + feature list) ----
@st.cache_resource
def load_model():
    return joblib.load("fraud_model.pkl")

bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
threshold = bundle["threshold"]
best_model_name = bundle.get("best_model_name", "Logistic Regression")

# Pull the feature list from the bundle itself instead of hardcoding it here,
# so the app always matches whatever the notebook was last trained with.
FEATURE_COLS = bundle["feature_cols"]

# Only Logistic Regression was trained on scaled features - tree models (Random
# Forest / XGBoost) were trained on the raw values, so scaling must be conditional.
def predict_proba(X):
    if best_model_name == "Logistic Regression":
        return model.predict_proba(scaler.transform(X))[:, 1]
    return model.predict_proba(X)[:, 1]

st.title("Healthcare Provider Fraud Detection")
st.caption(f"Model: {best_model_name} | Decision threshold: {threshold:.2f}")

tab1, tab2 = st.tabs(["Batch scoring (CSV)", "Single provider (manual entry)"])

# ---------------- Tab 1: Batch scoring ----------------
with tab1:
    st.subheader("Upload provider-level features")
    st.write(f"CSV must contain a `Provider` column plus these {len(FEATURE_COLS)} feature columns:")
    st.code(", ".join(FEATURE_COLS))

    file = st.file_uploader("Upload CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            X = df[FEATURE_COLS]
            probs = predict_proba(X)
            df["Probability"] = probs.round(4)
            df["PredictedClass"] = ["Yes" if p > threshold else "No" for p in probs]

            st.success(f"Scored {len(df)} providers | {(df['PredictedClass']=='Yes').sum()} flagged as fraud")
            st.dataframe(df.sort_values("Probability", ascending=False), use_container_width=True)

            st.download_button(
                "Download results as CSV",
                df.to_csv(index=False).encode("utf-8"),
                "fraud_predictions.csv",
                "text/csv",
            )

# ---------------- Tab 2: Single provider ----------------
with tab2:
    st.subheader("Enter a provider's features")
    col1, col2 = st.columns(2)
    with col1:
        total_claims = st.number_input("Total Claims", min_value=0, value=50)
        total_reimbursed = st.number_input("Total Reimbursed ($)", min_value=0.0, value=20000.0)
        avg_reimbursed = st.number_input("Avg Reimbursed ($)", min_value=0.0, value=400.0)
        unique_patients = st.number_input("Unique Patients", min_value=1, value=40)
        unique_physicians = st.number_input("Unique Physicians", min_value=1, value=10)
        inpatient_claims = st.number_input("Inpatient Claims", min_value=0, value=5)
        avg_length_of_stay = st.number_input("Avg Length Of Stay (days)", min_value=0.0, value=0.0)
    with col2:
        avg_patient_age = st.number_input("Avg Patient Age", min_value=0.0, value=72.0)
        avg_chronic = st.number_input("Avg Chronic Conditions", min_value=0.0, value=4.0)
        claims_per_patient = st.number_input("Claims Per Patient", min_value=0.0, value=1.25)
        reimbursed_per_patient = st.number_input("Reimbursed Per Patient ($)", min_value=0.0, value=500.0)
        inpatient_ratio = st.number_input("Inpatient Ratio", min_value=0.0, max_value=1.0, value=0.1)
        claims_per_physician = st.number_input("Claims Per Physician", min_value=0.0, value=5.0)
        avg_diagnosis_codes = st.number_input("Avg Diagnosis Codes", min_value=0.0, value=3.0)

    if st.button("Predict"):
        # Build by name, then reorder to FEATURE_COLS - stays correct even if the
        # feature list changes or is in a different order than the form above.
        values = {
            "TotalClaims": total_claims,
            "TotalReimbursed": total_reimbursed,
            "AvgReimbursed": avg_reimbursed,
            "UniquePatients": unique_patients,
            "UniquePhysicians": unique_physicians,
            "InpatientClaims": inpatient_claims,
            "AvgPatientAge": avg_patient_age,
            "AvgChronicConditions": avg_chronic,
            "ClaimsPerPatient": claims_per_patient,
            "ReimbursedPerPatient": reimbursed_per_patient,
            "InpatientRatio": inpatient_ratio,
            "ClaimsPerPhysician": claims_per_physician,
            "AvgLengthOfStay": avg_length_of_stay,
            "AvgDiagnosisCodes": avg_diagnosis_codes,
        }
        missing = [c for c in FEATURE_COLS if c not in values]
        if missing:
            st.error(f"This form doesn't collect: {missing}. Update the form to match the model's feature list.")
        else:
            row = pd.DataFrame([values])[FEATURE_COLS]
            prob = predict_proba(row)[0]
            pred = "Yes" if prob > threshold else "No"

            st.metric("Fraud Probability", f"{prob:.1%}")
            if pred == "Yes":
                st.error(f"Prediction: Potential Fraud (probability {prob:.1%} > threshold {threshold:.0%})")
            else:
                st.success(f"Prediction: No Fraud (probability {prob:.1%} <= threshold {threshold:.0%})")