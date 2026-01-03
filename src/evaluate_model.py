import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def main() -> None:
    df = pd.read_parquet("data/processed/features.parquet")
    y = df["resolution_hours"]
    X = df.drop(columns=["resolution_hours"])

    pipe = joblib.load("models/model.joblib")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preds = pipe.predict(X_test)

    os.makedirs("reports/figures", exist_ok=True)
    plt.figure()
    plt.hist(preds, bins=50)
    plt.title("Predicted resolution time distribution")
    plt.xlabel("Hours")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("reports/figures/predicted_resolution_hist.png", dpi=200)
    plt.close()

    print("Saved reports/figures/predicted_resolution_hist.png")

if __name__ == "__main__":
    main()
