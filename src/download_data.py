import os
import requests
import pandas as pd

API_URL = "https://data.cityofnewyork.us/resource/erm2-nwe9.json"

def download(limit: int = 50000) -> pd.DataFrame:
    params = {"$limit": limit, "$order": "created_date DESC"}
    r = requests.get(API_URL, params=params, timeout=60)
    r.raise_for_status()
    return pd.DataFrame(r.json())

def main() -> None:
    os.makedirs("data/raw", exist_ok=True)
    df = download(limit=50000)
    path = "data/raw/nyc_311_recent.parquet"
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path}")

if __name__ == "__main__":
    main()
