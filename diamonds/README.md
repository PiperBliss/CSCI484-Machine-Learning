# Diamond Data Analysis Project

This project analyzes the diamonds dataset using various machine learning and visualization techniques. The main files and their purposes are described below.

## File Overview

### 1. `diamonds.csv/diamonds.csv`
- **Description:** The main dataset containing diamond features and prices.
- **Location:** `diamonds.csv/diamonds.csv`
- **Usage:** Used as input for all scripts.

### 2. `diamond_eda.py`
- **Purpose:** Exploratory Data Analysis (EDA) and Random Forest Regression with feature subset search.
- **Key Features:**
  - Loads and cleans the diamonds dataset.
  - Encodes categorical features.
  - Performs EDA (summary statistics, correlations, etc.).
  - Trains a Random Forest Regressor on all feature combinations.
  - Evaluates models using wMAPE, RMSE, MAE, and R².
  - Saves results to `combinatorial_results.csv`.

### 3. `diamond_lr.py`
- **Purpose:** Linear Regression with combinatorial feature selection.
- **Key Features:**
  - Loads and cleans the diamonds dataset.
  - Encodes categorical features.
  - Trains a Linear Regression model on all feature combinations.
  - Evaluates models using wMAPE, RMSE, MAE, and R².
  - Saves results to `lr_combinatorial_results.csv`.

### 4. `diamonds_eda_rfr_wmape.py`
- **Purpose:** Advanced EDA and Random Forest Regression with exhaustive feature subset search (best by wMAPE).
- **Key Features:**
  - Detailed EDA (missing values, duplicates, value counts, plots).
  - Exhaustive search for the best feature subset using Random Forest.
  - Outputs EDA summaries, model metrics, and plots to the `outputs_diamonds/` directory.

### 5. `graphs.py`
- **Purpose:** Visualization of EDA and model results.
- **Key Features:**
  - Generates and saves plots: price distribution, carat vs price, price by cut, correlation heatmap.
  - Plots model performance (wMAPE vs number of features) using results from previous scripts.
  - Requires `combinatorial_results.csv` and/or `lr_combinatorial_results.csv` for model comparison plots.

---

## How to Run

1. **Activate your virtual environment:**
   ```
   source .venv/bin/activate
   ```

2. **Run EDA and modeling scripts:**
   ```
   python diamond_eda.py
   python diamond_lr.py
   python diamonds_eda_rfr_wmape.py
   ```

3. **Run visualization script:**
   ```
   python graphs.py
   ```

## Requirements

- Python 3.12+
- pandas, numpy, seaborn, matplotlib, scikit-learn

Install requirements with:
```
pip install pandas numpy seaborn matplotlib scikit-learn
```

---

## Notes

- Ensure `diamonds.csv/diamonds.csv` exists and is accessible.
- Output CSVs and plots will be saved in the project root or specified output folders.
- Do **not** place scripts or data inside the `venv/` or `.venv/` folders.

---

Let me know if you need further customization!
