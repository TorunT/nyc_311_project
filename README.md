# NYC 311 Service Requests: Resolution Time Prediction

This project downloads a recent sample of NYC 311 service requests from NYC Open Data, builds features, and trains a model to predict resolution time in hours.

## Setup (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

## Run
python src/download_data.py
python src/clean_features.py
python src/train_model.py
python src/evaluate_model.py

## Outputs
- data/raw: downloaded dataset
- data/processed: features table
- models/model.joblib: trained model pipeline
- reports/figures: charts for presentation
