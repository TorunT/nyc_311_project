import os
import pandas as pd

def to_datetime_safe(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["created_date"] = to_datetime_safe(df.get("created_date"))
    df["closed_date"] = to_datetime_safe(df.get("closed_date"))

    df["resolution_hours"] = (df["closed_date"] - df["created_date"]).dt.total_seconds() / 3600.0
    df = df[df["resolution_hours"].notna()]
    df = df[(df["resolution_hours"] >= 0) & (df["resolution_hours"] <= 24 * 30)]

    df["hour"] = df["created_date"].dt.hour
    df["day_of_week"] = df["created_date"].dt.dayofweek
    df["month"] = df["created_date"].dt.month

    df["complaint_type"] = df.get("complaint_type").fillna("Unknown")
    df["borough"] = df.get("borough").fillna("Unknown")

    keep = ["resolution_hours", "hour", "day_of_week", "month", "complaint_type", "borough"]
    return df[keep]

def main() -> None:
    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_parquet("data/raw/nyc_311_recent.parquet")
    out = build_features(df)
    path = "data/processed/features.parquet"
    out.to_parquet(path, index=False)
    print(f"Saved {len(out)} rows to {path}")

if __name__ == "__main__":
    main()
