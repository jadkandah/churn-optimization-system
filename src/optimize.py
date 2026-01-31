import pandas as pd
import joblib
from pathlib import Path


# Paths
DATA_PATH = Path("data/processed/churn_clean.csv")
MODEL_PATH = Path("models/logistic.pkl")
OUTPUT_PATH = Path("reports")
OUTPUT_PATH.mkdir(exist_ok=True)


# Business parameters
DISCOUNT = 50        # $ per customer
RETENTION_RATE = 0.4 # probability discount keeps customer
BUDGET = 10000       # total budget


def optimize():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Churn", axis=1)

    model = joblib.load(MODEL_PATH)

    # Predict churn probability
    churn_prob = model.predict_proba(X)[:, 1]

    df["churn_prob"] = churn_prob

    # Expected value if retained
    df["expected_revenue"] = df["MonthlyCharges"] * 12

    # Expected profit from targeting
    df["expected_profit"] = (
        df["churn_prob"]
        * RETENTION_RATE
        * df["expected_revenue"]
        - DISCOUNT
    )

    # Sort by best profit
    df = df.sort_values("expected_profit", ascending=False)

    max_customers = BUDGET // DISCOUNT

    selected = df.head(max_customers)

    selected.to_csv(OUTPUT_PATH / "optimized_targets.csv", index=False)

    print(f"Targeted customers: {len(selected)}")
    print(f"Expected total profit: ${selected['expected_profit'].sum():.2f}")
    print("Saved to reports/optimized_targets.csv")


if __name__ == "__main__":
    optimize()
