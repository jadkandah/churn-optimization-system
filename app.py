import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------
# Paths
# --------------------
DATA_PATH = Path("data/processed/churn_clean.csv")
MODEL_PATH = Path("models/logistic.pkl")
SHAP_PNG_PATH = Path("reports/shap_summary.png")
TARGETS_PATH = Path("reports/optimized_targets.csv")


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_targets():
    if TARGETS_PATH.exists():
        return pd.read_csv(TARGETS_PATH)
    return None


def build_input_form(feature_cols):
    st.subheader("Predict churn for a customer")
    st.caption("Fill the key fields. The app will automatically align inputs to the model’s one-hot encoded features.")

    with st.form("predict_form"):
        # Layout
        colA, colB = st.columns(2)

        with colA:
            st.markdown("#### Billing & Tenure")
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0, step=1.0)
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=20000.0, value=1000.0, step=10.0)

            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment Method",
                ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]
            )

        with colB:
            st.markdown("#### Services")
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])

        submitted = st.form_submit_button("Predict")

    # Build a single-row dict aligned to one-hot columns (default 0)
    row = {c: 0 for c in feature_cols}
    row["tenure"] = tenure
    row["MonthlyCharges"] = monthly
    row["TotalCharges"] = total

    # InternetService: baseline is DSL (drop_first=True)
    if internet == "Fiber optic" and "InternetService_Fiber optic" in row:
        row["InternetService_Fiber optic"] = 1
    elif internet == "No" and "InternetService_No" in row:
        row["InternetService_No"] = 1

    # Contract: baseline is Month-to-month
    if contract == "One year" and "Contract_One year" in row:
        row["Contract_One year"] = 1
    elif contract == "Two year" and "Contract_Two year" in row:
        row["Contract_Two year"] = 1

    # PaperlessBilling: baseline is No
    if paperless == "Yes" and "PaperlessBilling_Yes" in row:
        row["PaperlessBilling_Yes"] = 1

    # PaymentMethod: depends on one-hot presence; keep safe checks
    if payment == "Electronic check" and "PaymentMethod_Electronic check" in row:
        row["PaymentMethod_Electronic check"] = 1
    if payment == "Mailed check" and "PaymentMethod_Mailed check" in row:
        row["PaymentMethod_Mailed check"] = 1
    if payment == "Credit card (automatic)" and "PaymentMethod_Credit card (automatic)" in row:
        row["PaymentMethod_Credit card (automatic)"] = 1
    # Bank transfer (automatic) -> keep all PaymentMethod_* at 0 (baseline)

    # StreamingTV: baseline is No
    if streaming_tv == "Yes" and "StreamingTV_Yes" in row:
        row["StreamingTV_Yes"] = 1

    # StreamingMovies: baseline is No
    if streaming_movies == "Yes" and "StreamingMovies_Yes" in row:
        row["StreamingMovies_Yes"] = 1

    # TechSupport: baseline is No
    if tech_support == "Yes" and "TechSupport_Yes" in row:
        row["TechSupport_Yes"] = 1

    # OnlineSecurity: baseline is No
    if online_security == "Yes" and "OnlineSecurity_Yes" in row:
        row["OnlineSecurity_Yes"] = 1

    # MultipleLines: baseline is No
    if multiple_lines == "Yes" and "MultipleLines_Yes" in row:
        row["MultipleLines_Yes"] = 1

    df_one = pd.DataFrame([row])
    return submitted, df_one



def main():
    st.set_page_config(page_title="Churn Optimization System", layout="wide")
    st.title("Customer Churn Prediction & Optimization System")

    df = load_data()
    model = load_model()

    feature_cols = df.drop("Churn", axis=1).columns.tolist()

    tab1, tab2, tab3 = st.tabs(["Overview", "Explainability (SHAP)", "Optimization & Predict"])

    with tab1:
        st.subheader("Dataset snapshot")
        st.write(df.head())

        st.subheader("Model Selected:")
        st.write("Logistic Regression (scaled pipeline)")

        st.subheader("Latest evaluation (from your run)")
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", "0.8357")
        c2.metric("F1 (Churn)", "0.6091")
        c3.metric("Recall (Churn)", "0.5749")

    with tab2:
        st.subheader("Explainability (SHAP)")

        if SHAP_PNG_PATH.exists():

            # Small preview (centered)
            left, mid, right = st.columns([1, 2, 1])

            with mid:
                st.caption("Top features driving churn risk (preview)")
                st.image(str(SHAP_PNG_PATH), width=600)

            # Full-size version
            with st.expander("View full-size SHAP plot"):
                st.caption("Full resolution SHAP summary")
                st.image(str(SHAP_PNG_PATH), use_container_width=True)

        else:
            st.warning("reports/shap_summary.png not found. Run: python src/explain.py")

        st.divider()

        st.subheader("Key insights")

        c1, c2, c3 = st.columns(3)

        c1.info("Low tenure → higher churn risk")
        c2.info("High monthly charges → higher churn risk")
        c3.info("Long-term contracts → lower churn risk")

        st.caption(
            "These insights guide retention strategies such as onboarding programs, pricing adjustments, and contract incentives."
        )


    with tab3:
        st.subheader("Optimization output")
        targets = load_targets()
        if targets is not None:
            st.write("Top 20 recommended targets (highest expected profit):")
            st.dataframe(targets.head(20), use_container_width=True)
        else:
            st.warning("reports/optimized_targets.csv not found. Run: python src/optimize.py")

        st.divider()

        submitted, customer_df = build_input_form(feature_cols)

        if submitted:
            prob = model.predict_proba(customer_df)[0][1]

            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Churn probability", f"{prob:.2%}")

            with c2:
                if prob >= 0.6:
                    st.error("High risk: prioritize retention offer + support follow-up.")
                elif prob >= 0.4:
                    st.warning("Medium risk: consider lighter retention action.")
                else:
                    st.success("Lower risk: no immediate action needed.")


if __name__ == "__main__":
    main()
# To run this app, use the command:
# streamlit run app.py