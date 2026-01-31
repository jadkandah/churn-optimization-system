import pandas as pd
import joblib
from pathlib import Path


# Paths
MODEL_PATH = Path("models/logistic.pkl")
FEATURES_PATH = Path("data/processed/churn_clean.csv")


def load_features():
    df = pd.read_csv(FEATURES_PATH)
    return df.drop("Churn", axis=1).columns


def predict_single(customer_data: dict):
    """
    customer_data: dictionary of feature values
    """

    # Load model
    model = joblib.load(MODEL_PATH)

    # Load expected columns
    columns = load_features()

    # Create dataframe
    df = pd.DataFrame([customer_data])

    # Align columns
    df = df.reindex(columns=columns, fill_value=0)

    # Predict
    prob = model.predict_proba(df)[0][1]

    return prob


def demo():
    # Example customer (high risk)
    example_customer = {
        "tenure": 2,
        "MonthlyCharges": 95,
        "TotalCharges": 180,
        "InternetService_Fiber optic": 1,
        "Contract_Two year": 0,
        "Contract_One year": 0,
        "StreamingTV_Yes": 1,
        "StreamingMovies_Yes": 1,
        "PaperlessBilling_Yes": 1,
        "PaymentMethod_Electronic check": 1
    }

    prob = predict_single(example_customer)

    print(f"Churn Probability: {prob:.2%}")


if __name__ == "__main__":
    demo()
