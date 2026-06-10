import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import joblib

# 1. Load Data
file_path = "Australian_job_data_cleaned.csv"
print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

# 2. Feature Selection
# Drop target-derived columns (Leakage) and redundant meta-data
drop_cols = [
    'salary_min_aud', 'salary_max_aud', 'salary_range_aud', 
    'salary_log_median', 'salary_bracket', 'salary_variability',
    'city_salary_adjusted_index', # High correlation (0.96) indicates leakage/target-encoding
    'year', 'month_name', 'remote_availability', 'quarter'
]

# Separate features and target
X = df.drop(columns=drop_cols + ['salary_median_aud'])
y = df['salary_median_aud']

# Handle potentially remaining non-numeric columns if any (e.g. 'demand_index' if it was parsed as string)
# Based on EDA, most seem numeric or encoded. checking for object types just in case.
object_cols = X.select_dtypes(include=['object']).columns
print(f"Categorical columns to encoding (if any remain): {object_cols}")

# Convert 'Yes'/'No' to 1/0 if needed (e.g. is_high_demand might be string or int)
# df inspection showed 'is_high_demand' is 1/0. 'remote_availability' is Yes/No but we dropped it for 'remote_flag'
# Checking other object cols just in case.
for col in X.select_dtypes(include=['object']).columns:
    try:
        X[col] = pd.to_numeric(X[col])
    except:
        # If simple conversion fails, use get_dummies or factorize -> For now assuming they are mostly clean based on "cleaned" in filename
        print(f"Column {col} is object, applying OneHot encoding or drop...")
        X = pd.get_dummies(X, columns=[col], drop_first=True)

print(f"Features selected: {X.columns.tolist()}")

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Pipeline Construction
# Random Forest is robust, but scaling helps if we introduce other models later.
# Imputing just in case of missing values
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
])

# 5. Training
print("Training model...")
pipeline.fit(X_train, y_train)

# 6. Evaluation
print("Evaluating model...")
y_pred = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("\n--- Model Performance Metrics ---")
print(f"MAE:  {mae:,.2f}")
print(f"MSE:  {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R2 :  {r2:.4f}")
print(f"MAPE: {mape:.2%}")

# 7. Save Model
model_filename = 'model_pipeline.joblib'
joblib.dump(pipeline, model_filename)
print(f"\nModel saved to {model_filename}")

# Save feature names for App usage
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'model_features.joblib')
print("Model features list saved to model_features.joblib")
