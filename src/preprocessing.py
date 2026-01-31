import pandas as pd
from pathlib import Path


# Paths
RAW_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_PATH = Path("data/processed/churn_clean.csv")


def preprocess():
    # Load data
    df = pd.read_csv(RAW_PATH)

    # Convert TotalCharges to numeric (has spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop missing values
    df = df.dropna()

    # Drop customerID (no predictive value)
    df = df.drop(columns=["customerID"])

    # Convert target to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encode categorical features
    df = pd.get_dummies(df, drop_first=True)

    # Save clean data
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)

    print("Preprocessing complete.")
    print(f"Saved to {PROCESSED_PATH}")


if __name__ == "__main__":
    preprocess()
