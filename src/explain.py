import pandas as pd
import joblib
import shap
from pathlib import Path
import matplotlib.pyplot as plt


# Paths
DATA_PATH = Path("data/processed/churn_clean.csv")
MODEL_PATH = Path("models/logistic.pkl")
OUTPUT_PATH = Path("reports")
OUTPUT_PATH.mkdir(exist_ok=True)


def explain():
    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Churn", axis=1)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Extract classifier from pipeline
    clf = model.named_steps["clf"]
    scaler = model.named_steps["scaler"]

    # Scale data
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns
    )


    # SHAP explainer
    explainer = shap.Explainer(clf, X_scaled)
    shap_values = explainer(X_scaled)

    # Summary plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        show=False
    )

    plt.savefig(OUTPUT_PATH / "shap_summary.png", dpi=300, bbox_inches="tight")
    print("SHAP explanation saved to reports/shap_summary.png")


if __name__ == "__main__":
    explain()
