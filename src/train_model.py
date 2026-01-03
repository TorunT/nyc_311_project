import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

def main() -> None:
    df = pd.read_parquet("data/processed/features.parquet")
    y = df["resolution_hours"]
    X = df.drop(columns=["resolution_hours"])

    cat_cols = ["complaint_type", "borough"]
    num_cols = ["hour", "day_of_week", "month"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )

    pipe = Pipeline(steps=[("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipe, "models/model.joblib")

    print("Model saved to models/model.joblib")
    print(f"MAE in hours: {mae:.3f}")

if __name__ == "__main__":
    main()
