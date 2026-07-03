import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

# Plot 1: Price Distribution (Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Diamond Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.savefig('eda_price_hist.png')
plt.close()

# Plot 2: Carat vs Price (Scatter)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df.sample(2000, random_state=42), x='carat', y='price', hue='clarity', alpha=0.5, palette='viridis')
plt.title('Carat vs Price (Colored by Clarity)')
plt.xlabel('Carat Weight')
plt.ylabel('Price ($)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('eda_carat_price_scatter.png')
plt.close()

# Plot 3: Price by Cut (Violin Plot for Readability)
plt.figure(figsize=(12, 7))
sns.violinplot(data=df, x='cut', y='price', order=cut_order, palette='Set2', inner="quartile")
plt.title('Density Distribution of Price by Cut Quality (Violin Plot)')
plt.xlabel('Cut Quality')
plt.ylabel('Price ($)')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.savefig('eda_price_by_cut_violin.png')
plt.close()

# Plot 4: Correlation Heatmap
plt.figure(figsize=(12, 10))
cols_to_corr = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z', 'cut_num', 'color_num', 'clarity_num']
mask = np.triu(np.ones_like(df[cols_to_corr].corr(), dtype=bool))
sns.heatmap(df[cols_to_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.savefig('eda_heatmap.png')
plt.close()


# ==========================================
# 3. GENERATE MODEL COMPARISON PLOTS (EXPANDED)
# ==========================================
print("Generating Model Comparison Plots from CSVs...")
try:
    # Load the combinatorial results
    rfr_results = pd.read_csv('combinatorial_results.csv')
    lr_results = pd.read_csv('lr_combinatorial_results.csv')

    # Plot 5: wMAPE vs Number of Features
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=rfr_results, x='num_features', y='wMAPE', marker='o', estimator='min', errorbar=None, label='Random Forest (RFR)', color='blue')
    sns.lineplot(data=lr_results, x='num_features', y='wMAPE', marker='s', estimator='min', errorbar=None, label='Linear Regression (LR)', color='green')
    plt.title('Algorithm Comparison: Best wMAPE vs Number of Features')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Best wMAPE (Lower is Better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_wmape_comparison.png')
    plt.close()

    # Plot 6: R² Score vs Number of Features
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=rfr_results, x='num_features', y='R2', marker='o', estimator='max', errorbar=None, label='Random Forest (RFR)', color='blue')
    sns.lineplot(data=lr_results, x='num_features', y='R2', marker='s', estimator='max', errorbar=None, label='Linear Regression (LR)', color='green')
    plt.title('Algorithm Comparison: Best R² Score vs Number of Features')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Best R² Score (Higher is Better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_r2_comparison.png')
    plt.close()

    # Plot 7: RMSE vs Number of Features
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=rfr_results, x='num_features', y='RMSE', marker='o', estimator='min', errorbar=None, label='Random Forest (RFR)', color='blue')
    sns.lineplot(data=lr_results, x='num_features', y='RMSE', marker='s', estimator='min', errorbar=None, label='Linear Regression (LR)', color='green')
    plt.title('Algorithm Comparison: Best RMSE vs Number of Features')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Best RMSE (Lower is Better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_rmse_comparison.png')
    plt.close()

    # Plot 8: MAE vs Number of Features
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=rfr_results, x='num_features', y='MAE', marker='o', estimator='min', errorbar=None, label='Random Forest (RFR)', color='blue')
    sns.lineplot(data=lr_results, x='num_features', y='MAE', marker='s', estimator='min', errorbar=None, label='Linear Regression (LR)', color='green')
    plt.title('Algorithm Comparison: Best MAE vs Number of Features')
    plt.xlabel('Number of Features Selected')
    plt.ylabel('Best MAE (Lower is Better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_mae_comparison.png')
    plt.close()

    # Plot 9: Overall Best Metrics Bar Chart (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Min/Max functions applied to find overall best performances
    best_rfr_wmape = rfr_results['wMAPE'].min()
    best_lr_wmape = lr_results['wMAPE'].min()
    best_rfr_r2 = rfr_results['R2'].max()
    best_lr_r2 = lr_results['R2'].max()
    best_rfr_rmse = rfr_results['RMSE'].min()
    best_lr_rmse = lr_results['RMSE'].min()
    best_rfr_mae = rfr_results['MAE'].min()
    best_lr_mae = lr_results['MAE'].min()

    # Top Left: wMAPE
    sns.barplot(x=['Random Forest', 'Linear Regression'], y=[best_rfr_wmape, best_lr_wmape], palette='Set1', ax=axes[0,0])
    axes[0,0].set_title('Overall Best wMAPE (Lower is Better)')
    axes[0,0].set_ylabel('wMAPE')
    axes[0,0].bar_label(axes[0,0].containers[0], fmt='%.4f')

    # Top Right: R² Score
    sns.barplot(x=['Random Forest', 'Linear Regression'], y=[best_rfr_r2, best_lr_r2], palette='Set2', ax=axes[0,1])
    axes[0,1].set_title('Overall Best R² Score (Higher is Better)')
    axes[0,1].set_ylabel('R² Score')
    axes[0,1].bar_label(axes[0,1].containers[0], fmt='%.4f')

    # Bottom Left: RMSE
    sns.barplot(x=['Random Forest', 'Linear Regression'], y=[best_rfr_rmse, best_lr_rmse], palette='Set3', ax=axes[1,0])
    axes[1,0].set_title('Overall Best RMSE (Lower is Better)')
    axes[1,0].set_ylabel('RMSE ($)')
    axes[1,0].bar_label(axes[1,0].containers[0], fmt='%.2f')

    # Bottom Right: MAE
    sns.barplot(x=['Random Forest', 'Linear Regression'], y=[best_rfr_mae, best_lr_mae], palette='Pastel1', ax=axes[1,1])
    axes[1,1].set_title('Overall Best MAE (Lower is Better)')
    axes[1,1].set_ylabel('MAE ($)')
    axes[1,1].bar_label(axes[1,1].containers[0], fmt='%.2f')

    plt.tight_layout()
    plt.savefig('model_all_metrics_bar_chart.png')
    plt.close()

    print("All comparison plots generated successfully!")

except FileNotFoundError as e:
    print(f"Skipping Model Comparison: Could not find required CSV files. Error: {e}")
except KeyError as e:
    print(f"Data missing! Ensure your RFR and LR scripts track {e}. You may need to re-run your model training.")

print("Done. Check your folder for the generated .png files.")