import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_wmape(y_true, y_pred):
    """Calculates Weighted Mean Absolute Percentage Error."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

# 1. LOAD AND CLEAN DATA
print("Loading data...")
df = pd.read_csv('diamonds.csv/diamonds.csv').drop(columns=['Unnamed: 0'], errors='ignore')

# Identify and remove rows with invalid zero dimensions
df = df[(df[['x','y','z']] != 0).all(axis=1)]

# Encode Ordinal Categorical Features
cut_map = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_map = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
clarity_map = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

df['cut'] = df['cut'].map(cut_map)
df['color'] = df['color'].map(color_map)
df['clarity'] = df['clarity'].map(clarity_map)

# 2. COMBINATORIAL EVALUATION
target = 'price'
features = [col for col in df.columns if col != target]
results = []

# Using a sample for the search to maintain speed (increase n if you want more precision)
df_sample = df.sample(n=3000, random_state=42) 

print("Starting Random Forest Combinatorial Evaluation (511 combinations)...")
print("This may take a few minutes...")

for r in range(1, len(features) + 1):
    for combo in combinations(features, r):
        X = df_sample[list(combo)]
        y = df_sample[target]
        
        # Split into train, validation, and test sets (60/20/20)
        X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Log metrics for validation and test sets
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

results_df = pd.DataFrame(results)
best_model = results_df.loc[results_df['wMAPE_val'].idxmin()]

print("\n--- BEST COMBINATION BY wMAPE (Random Forest, Validation Set) ---")
print(f"Features: {best_model['features']}")
print(f"wMAPE (val): {best_model['wMAPE_val']:.4f}")
print(f"RMSE (val): {best_model['RMSE_val']:.4f}")
print(f"MAE (val): {best_model['MAE_val']:.4f}")
print(f"R2 Score (val): {best_model['R2_val']:.4f}")
print(f"wMAPE (test): {best_model['wMAPE_test']:.4f}")
print(f"RMSE (test): {best_model['RMSE_test']:.4f}")
print(f"MAE (test): {best_model['MAE_test']:.4f}")
print(f"R2 Score (test): {best_model['R2_test']:.4f}")

# Save full results
results_df.to_csv('combinatorial_results.csv', index=False)
print("\nSuccess! Saved to 'combinatorial_results.csv'. You can now run your graphs script.")