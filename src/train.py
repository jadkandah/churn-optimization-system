import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib


# Paths
DATA_PATH = Path("data/processed/churn_clean.csv")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)


def train():
    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "logistic": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000))
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print(f"{name} ROC-AUC: {auc:.4f}")
        print(f"{name} F1-score: {f1:.4f}")
        print(f"{name} Recall (Churn): {recall:.4f}")
        print(classification_report(y_test, y_pred))

        # Save model
        joblib.dump(model, MODEL_PATH / f"{name}.pkl")

        results[name] = auc

    print("\nTraining complete.")
    print("Results:", results)


if __name__ == "__main__":
    train()
