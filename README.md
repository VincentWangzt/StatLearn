# Statistical Learning Based Pediatric Appendicitis Prediction

## Project Overview
This project applies statistical learning techniques to predict pediatric appendicitis diagnosis, severity, and management decisions. Using the Regensburg Pediatric Appendicitis Dataset, we develop machine learning models that focus on a clinically practical scenario where ultrasound imaging is unavailable. By utilizing only non-ultrasound features (demographics, physical exam, lab tests), we aim to create robust diagnostic tools for resource-limited environments.

## Authors
- **Zitong Wang** (School of Mathematical Sciences, Peking University)
- **Wen Yuan** (School of Mathematical Sciences, Peking University)
- **Jingxing Zhou** (School of Mathematical Sciences, Peking University)

## Dataset
The project uses the **Regensburg Pediatric Appendicitis Dataset** (UCI Machine Learning Repository), comprising 782 patients with 58 features.
- **Source:** Children's Hospital St. Hedwig, Regensburg, Germany (2016-2021).
- **Focus:** We primarily use a subset of 45 features, excluding ultrasound-derived metrics to simulate settings without advanced imaging capabilities.

## Project Structure
```
project/
├── data/               # Raw dataset (app_data.xlsx) and documentation
├── processed_data/     # Preprocessed numpy arrays (X_train, y_train, etc.)
├── models/             # Model training results (CSVs)
├── analysis/           # Exploratory data analysis outputs and plots
├── report/             # LaTeX source for the project report
├── preprocess.py       # Data cleaning, encoding, and splitting
├── analysis.py         # Exploratory data analysis and visualization
├── model_training.py   # Model training, evaluation, and feature selection
└── README.md           # Project documentation
```

## Requirements
The project requires a Python environment with the following libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `openpyxl` (for reading Excel files)

## Usage

### 1. Data Preprocessing
Run the preprocessing script to clean the data, handle missing values, and generate training/testing splits.
```bash
python preprocess.py
```
This will create `.npy` files in the `processed_data/` directory.

### 2. Exploratory Data Analysis
Generate correlation plots and other visualizations to understand the data distribution.
```bash
python analysis.py
```
Outputs will be saved in the `analysis/` directory.

### 3. Model Training and Evaluation
Train various machine learning models (Logistic Regression, Random Forest, Gradient Boosting, etc.), perform feature selection, and evaluate performance.
```bash
python model_training.py
```
Results (CSV files) will be saved in the `models/` directory.

## Key Results
The project evaluates models on three tasks:
1.  **Diagnosis:** Appendicitis vs. No Appendicitis
2.  **Severity:** Complicated vs. Uncomplicated Appendicitis
3.  **Management:** Surgical vs. Conservative Management

Detailed performance metrics (Accuracy, AUC, etc.) can be found in the `models/` directory and the final report.
