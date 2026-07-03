import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# ==========================================
# 1. LOAD AND PREPARE ORIGINAL DATA (For EDA)
# ==========================================
print("Loading original diamond data for EDA...")
df = pd.read_csv('diamonds.csv/diamonds.csv').drop(columns=['Unnamed: 0'], errors='ignore')
df = df[(df[['x', 'y', 'z']] != 0).all(axis=1)] # Remove invalid rows

# Encode for modeling/heatmap
cut_map = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_map = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
clarity_map = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}
cut_order = ["Fair", "Good", "Very Good", "Premium", "Ideal"]

df['cut_num'] = df['cut'].map(cut_map)
df['color_num'] = df['color'].map(color_map)
df['clarity_num'] = df['clarity'].map(clarity_map)

# ==========================================
# 2. GENERATE EDA PLOTS
# ==========================================
print("Generating EDA Plots...")

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Diamond Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.savefig('eda_price_hist.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(2000, random_state=42), x='carat', y='price', hue='clarity', alpha=0.5, palette='viridis')
plt.title('Carat vs Price (Colored by Clarity)')
plt.xlabel('Carat Weight')
plt.ylabel('Price ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('eda_carat_price_scatter.png')
plt.close()

plt.figure(figsize=(12, 7))
sns.violinplot(data=df, x='cut', y='price', order=cut_order, palette='Set2', inner="quartile")
plt.title('Density Distribution of Price by Cut Quality (Violin Plot)')
plt.xlabel('Cut Quality')
plt.ylabel('Price ($)')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.savefig('eda_price_by_cut_violin.png')
plt.close()

plt.figure(figsize=(12, 10))
cols_to_corr = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z', 'cut_num', 'color_num', 'clarity_num']
mask = np.triu(np.ones_like(df[cols_to_corr].corr(), dtype=bool))
sns.heatmap(df[cols_to_corr].corr(), mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig('eda_heatmap.png')
plt.close()

# ==========================================
# 3. GENERATE MODEL COMPARISON PLOTS (VAL & TEST)
# ==========================================
print("Generating Model Comparison Plots from CSVs...")
try:
    # Load the combinatorial results
    rfr_results = pd.read_csv('combinatorial_results.csv')
    lr_results = pd.read_csv('lr_combinatorial_results.csv')

    # Helper function to plot line graphs easily
    def plot_metric_comparison(metric_name, title, ylabel, estimator, is_higher_better=False):
        plt.figure(figsize=(10, 6))
        # RFR Lines
        sns.lineplot(data=rfr_results, x='num_features', y=f'{metric_name}_val', marker='o', estimator=estimator, errorbar=None, label='RFR (Validation)', color='blue', linestyle='--')
        sns.lineplot(data=rfr_results, x='num_features', y=f'{metric_name}_test', marker='o', estimator=estimator, errorbar=None, label='RFR (Test)', color='blue')
        # LR Lines
        sns.lineplot(data=lr_results, x='num_features', y=f'{metric_name}_val', marker='s', estimator=estimator, errorbar=None, label='LR (Validation)', color='green', linestyle='--')
        sns.lineplot(data=lr_results, x='num_features', y=f'{metric_name}_test', marker='s', estimator=estimator, errorbar=None, label='LR (Test)', color='green')
        
        plt.title(title)
        plt.xlabel('Number of Features Selected')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'model_{metric_name.lower()}_comparison.png')
        plt.close()

    # Plot 5: wMAPE vs Number of Features
    plot_metric_comparison('wMAPE', 'Algorithm Comparison: Best wMAPE vs Number of Features', 'Best wMAPE (Lower is Better)', 'min')
    # Plot 6: R² Score vs Number of Features
    plot_metric_comparison('R2', 'Algorithm Comparison: Best R² Score vs Number of Features', 'Best R² Score (Higher is Better)', 'max', True)
    # Plot 7: RMSE vs Number of Features
    plot_metric_comparison('RMSE', 'Algorithm Comparison: Best RMSE vs Number of Features', 'Best RMSE (Lower is Better)', 'min')
    # Plot 8: MAE vs Number of Features
    plot_metric_comparison('MAE', 'Algorithm Comparison: Best MAE vs Number of Features', 'Best MAE (Lower is Better)', 'min')

    # Plot 9: Overall Best Metrics Bar Chart (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Compile Best Validation and Test Scores
    best_scores = pd.DataFrame({
        'Model': ['RFR', 'RFR', 'LR', 'LR'],
        'Dataset': ['Validation', 'Test', 'Validation', 'Test'],
        'wMAPE': [rfr_results['wMAPE_val'].min(), rfr_results['wMAPE_test'].min(), lr_results['wMAPE_val'].min(), lr_results['wMAPE_test'].min()],
        'R2': [rfr_results['R2_val'].max(), rfr_results['R2_test'].max(), lr_results['R2_val'].max(), lr_results['R2_test'].max()],
        'RMSE': [rfr_results['RMSE_val'].min(), rfr_results['RMSE_test'].min(), lr_results['RMSE_val'].min(), lr_results['RMSE_test'].min()],
        'MAE': [rfr_results['MAE_val'].min(), rfr_results['MAE_test'].min(), lr_results['MAE_val'].min(), lr_results['MAE_test'].min()]
    })

    # Top Left: wMAPE
    sns.barplot(data=best_scores, x='Model', y='wMAPE', hue='Dataset', palette='Set1', ax=axes[0,0])
    axes[0,0].set_title('Overall Best wMAPE (Lower is Better)')
    for container in axes[0,0].containers:
        axes[0,0].bar_label(container, fmt='%.4f', padding=3)

    # Top Right: R² Score
    sns.barplot(data=best_scores, x='Model', y='R2', hue='Dataset', palette='Set2', ax=axes[0,1])
    axes[0,1].set_title('Overall Best R² Score (Higher is Better)')
    for container in axes[0,1].containers:
        axes[0,1].bar_label(container, fmt='%.4f', padding=3)

    # Bottom Left: RMSE
    sns.barplot(data=best_scores, x='Model', y='RMSE', hue='Dataset', palette='Set3', ax=axes[1,0])
    axes[1,0].set_title('Overall Best RMSE (Lower is Better)')
    for container in axes[1,0].containers:
        axes[1,0].bar_label(container, fmt='%.2f', padding=3)

    # Bottom Right: MAE
    sns.barplot(data=best_scores, x='Model', y='MAE', hue='Dataset', palette='Pastel1', ax=axes[1,1])
    axes[1,1].set_title('Overall Best MAE (Lower is Better)')
    for container in axes[1,1].containers:
        axes[1,1].bar_label(container, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.savefig('model_all_metrics_bar_chart_val_test.png')
    plt.close()
    
    # ==========================================
    # 4. NEW: FEATURE IMPORTANCE BAR CHART
    # ==========================================
    print("Generating Feature Importance Bar Chart...")
    
    # We train a quick Random Forest on the full dataset specifically to extract feature importance
    features_imp = ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_num', 'color_num', 'clarity_num']
    rf_quick = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_quick.fit(df[features_imp], df['price'])

    importance_df = pd.DataFrame({
        'Feature': ['Carat', 'Depth', 'Table', 'Length (x)', 'Width (y)', 'Depth (z)', 'Cut', 'Color', 'Clarity'],
        'Importance': rf_quick.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Relative Importance (Contribution to Prediction)')
    plt.tight_layout()
    plt.savefig('model_feature_importance.png')
    plt.close()

    print("All comparison plots (Validation & Test) generated successfully!")

except FileNotFoundError as e:
    print(f"Skipping Model Comparison: Could not find required CSV files. Error: {e}")
except KeyError as e:
    print(f"Data missing! Ensure your RFR and LR scripts track {e}. You may need to re-run your model training.")

print("Done. Check your folder for the generated .png files.")