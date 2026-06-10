import pandas as pd
import numpy as np

# Load dataset
file_path = r"c:\Users\DELL\Downloads\Vcube Exercises\salary prediction on aus job market\Australian_job_data_cleaned.csv"
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

target = 'salary_median_aud'

# 1. Basic Info
print("\n--- Dataset Info ---")
print(f"Shape: {df.shape}")
print(df.dtypes)

# 2. Correlation Analysis
print("\n--- Correlation with Target ---")
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()[target].sort_values(ascending=False)

print(correlations)

# 3. Leakage Detection
print("\n--- Potential Leakage (Correlation > 0.8) ---")
high_corr = correlations[abs(correlations) > 0.8]
print(high_corr)

# 4. Low Importance
print("\n--- Low Importance (Correlation < 0.05) ---")
low_corr = correlations[abs(correlations) < 0.05]
print(low_corr)

# 5. Check for Redundancy (Text vs Encoded)
print("\n--- Columns List ---")
print(df.columns.tolist())
