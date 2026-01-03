# Report: NYC 311 Service Requests (Resolution Time Prediction)

## 1. Project Overview
This project uses public NYC 311 service request data to explore patterns in service requests and to build a baseline model that predicts the time it takes for a request to be resolved. The main goal is to demonstrate an end to end data science workflow: data ingestion, cleaning, feature engineering, modeling, evaluation, and communicating results in a reproducible way.

## 2. Data Source
**Dataset:** NYC 311 Service Requests  
**Provider:** NYC Open Data (public dataset)  
**Access method:** API download using a simple HTTP request

The project downloads a recent sample of service requests, then focuses on entries that have both a creation timestamp and a closed timestamp to compute a target variable for modeling.

## 3. Problem Definition
**Task:** Regression  
**Target variable:** Resolution time in hours

Resolution time is computed as:

- `resolution_hours = (closed_date - created_date)` in hours

The model attempts to predict resolution hours using features available at request creation time (time based features and categorical descriptors).

## 4. Data Preparation and Cleaning
### 4.1 Parsing timestamps
- `created_date` and `closed_date` are parsed as timezone aware timestamps.
- Rows with missing timestamps or invalid dates are removed.

### 4.2 Filtering unrealistic values
To reduce noise and extreme outliers:
- Only non negative resolution times are kept.
- Extremely large resolution times are filtered (for example, values greater than 30 days are excluded).

This filtering makes the baseline model more stable and the metric easier to interpret.

## 5. Feature Engineering
The following features are created:

### 5.1 Time features
- `hour` of day when the request was created
- `day_of_week` of creation (0 to 6)
- `month` of creation (1 to 12)

### 5.2 Categorical features
- `complaint_type` (filled with `Unknown` when missing)
- `borough` (filled with `Unknown` when missing)

Categorical variables are encoded using one hot encoding.

## 6. Modeling Approach
### 6.1 Baseline model
A baseline model was trained using:
- **RandomForestRegressor**

The modeling pipeline includes:
- preprocessing (one hot encoding for categorical features, pass through numerical features)
- model training
- evaluation using a train test split

### 6.2 Why this model
Random forests are a strong baseline for tabular data because they:
- handle non linear relationships
- are relatively robust to noise
- work well without heavy feature scaling

## 7. Evaluation
### 7.1 Metric
Model performance is measured using:
- **Mean Absolute Error (MAE)** in hours

MAE is easy to interpret: it represents the average absolute difference between predicted and true resolution time.

### 7.2 Result
**MAE (hours):** `ADD_YOUR_MAE_HERE`

Replace the placeholder above with the MAE printed by `python src/train_model.py`.

## 8. Results and Visualization
A simple visualization is included to show the distribution of predicted resolution times:

- `reports/figures/predicted_resolution_hist.png`

Add this to your README for presentation:

```md
![Predicted resolution distribution](reports/figures/predicted_resolution_hist.png)
