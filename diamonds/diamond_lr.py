import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_wmape(y_true, y_pred):
    """Calculates Weighted Mean Absolute Percentage Error."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

# 1. Load and Clean Data
df = pd.read_csv('diamonds.csv/diamonds.csv').drop(columns=['Unnamed: 0'], errors='ignore')

# Remove physically impossible rows (dimension 0)
df = df[(df[['x', 'y', 'z']] != 0).all(axis=1)]

# Encode Ordinal Categorical Features
cut_map = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_map = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
clarity_map = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

df['cut'] = df['cut'].map(cut_map)
df['color'] = df['color'].map(color_map)
df['clarity'] = df['clarity'].map(clarity_map)

# 2. Combinatorial Evaluation with Linear Regression
target = 'price'
features = [col for col in df.columns if col != target]
results = []

print("--- Starting Combinatorial Evaluation with Linear Regression ---")

# Iterate through every possible combination of features (2^9 - 1 = 511 combinations)
for r in range(1, len(features) + 1):
    for combo in combinations(features, r):
        X = df[list(combo)]
        y = df[target]
        
        # Split into train, validation, and test sets (60/20/20)
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        results.append({
            'num_features': r,
            'features': combo,
            'wMAPE_val': calculate_wmape(y_val, y_val_pred),
            'RMSE_val': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'MAE_val': mean_absolute_error(y_val, y_val_pred),
            'R2_val': r2_score(y_val, y_val_pred),
            'wMAPE_test': calculate_wmape(y_test, y_test_pred),
            'RMSE_test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'MAE_test': mean_absolute_error(y_test, y_test_pred),
            'R2_test': r2_score(y_test, y_test_pred)
        })

# Store results in a DataFrame and find the best based on wMAPE
results_df = pd.DataFrame(results)
best_model = results_df.loc[results_df['wMAPE_val'].idxmin()]

print("\n--- BEST COMBINATION BY wMAPE (Linear Regression, Validation Set) ---")
print(f"Features: {best_model['features']}")
print(f"Number of Features: {best_model['num_features']}")
print(f"wMAPE (val): {best_model['wMAPE_val']:.4f}")
print(f"RMSE (val): {best_model['RMSE_val']:.4f}")
print(f"MAE (val): {best_model['MAE_val']:.4f}")
print(f"R2 Score (val): {best_model['R2_val']:.4f}")
print(f"wMAPE (test): {best_model['wMAPE_test']:.4f}")
print(f"RMSE (test): {best_model['RMSE_test']:.4f}")
print(f"MAE (test): {best_model['MAE_test']:.4f}")
print(f"R2 Score (test): {best_model['R2_test']:.4f}")

# Save full results to CSV for further analysis
results_df.to_csv('lr_combinatorial_results.csv', index=False)