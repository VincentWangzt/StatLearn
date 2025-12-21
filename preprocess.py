"""
Data Preprocessing Script for Child Appendicitis ML Project
Faithfully records missing values (as NaN) rather than imputing them.
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Create output directories
os.makedirs('./processed_data', exist_ok=True)
os.makedirs('./analysis', exist_ok=True)

# Read the Excel file
print("Reading Excel file...")
df_data = pd.read_excel('./data/app_data.xlsx', sheet_name='All cases')
df_meta = pd.read_excel('./data/app_data.xlsx', sheet_name='Data Summary')

print(f"Original data shape: {df_data.shape}")
print(f"Columns: {df_data.columns.tolist()}")

# Identify target columns
target_columns = ['Diagnosis', 'Management', 'Severity']

# Identify Ultrasound category columns based on metadata
ultrasound_columns = [
    'US_Performed', 'US_Number', 'Appendix_on_US', 'Appendix_Diameter',
    'Free_Fluids', 'Appendix_Wall_Layers', 'Target_Sign', 'Appendicolith',
    'Perfusion', 'Perforation', 'Surrounding_Tissue_Reaction',
    'Appendicular_Abscess', 'Abscess_Location', 'Pathological_Lymph_Nodes',
    'Lymph_Nodes_Location', 'Bowel_Wall_Thickening',
    'Conglomerate_of_Bowel_Loops', 'Ileus', 'Coprostasis', 'Meteorism',
    'Enteritis', 'Gynecological_Findings'
]

# Columns to drop (free-form text or not useful for prediction)
columns_to_drop = [
    'Diagnosis_Presumptive',  # Free-form text, presumptive
    'Abscess_Location',  # Free-form text
    'Lymph_Nodes_Location',  # Free-form text (if exists)
    'Gynecological_Findings',  # Free-form text
]

# Dictionary to track transformations
transformations = {}
# Dictionary to track missing value counts
missing_value_stats = {}


def preprocess_data(df, include_ultrasound=True):
    """
    Preprocess the data for model training.
    Missing values are faithfully preserved as NaN.
    
    Args:
        df: DataFrame with raw data
        include_ultrasound: Whether to include ultrasound features
    
    Returns:
        X: Feature matrix (numpy array with NaN for missing values)
        y: Target matrix (numpy array)
        feature_names: List of feature names
        target_names: List of target names
    """
    df_work = df.copy()

    # Remove columns to drop
    for col in columns_to_drop:
        if col in df_work.columns:
            df_work = df_work.drop(columns=[col])
            transformations[col] = "Dropped (free-form text or not useful)"

    # If not including ultrasound, drop those columns
    if not include_ultrasound:
        for col in ultrasound_columns:
            if col in df_work.columns:
                df_work = df_work.drop(columns=[col])

    # Separate features and targets
    feature_cols = [c for c in df_work.columns if c not in target_columns]

    # Process each column
    processed_features = []
    feature_names = []

    for col in feature_cols:
        series = df_work[col]

        # Skip columns that are all NaN
        if series.isna().all():
            transformations[col] = "Dropped (all missing values)"
            continue

        # Record missing value statistics
        missing_count = series.isna().sum()
        missing_pct = (missing_count / len(series)) * 100
        missing_value_stats[col] = {
            'count': missing_count,
            'percentage': missing_pct
        }

        # Determine column type and process
        if series.dtype == 'object' or series.dtype.name == 'category':
            # Categorical column
            unique_vals = series.dropna().unique()

            if len(unique_vals) == 2:
                # Binary encoding - preserve NaN
                le = LabelEncoder()
                le.fit(unique_vals)  # Fit only on non-NaN values

                # Create encoded array preserving NaN
                encoded = np.full(len(series), np.nan)
                non_nan_mask = ~series.isna()
                encoded[non_nan_mask] = le.transform(series[non_nan_mask])

                processed_features.append(encoded.reshape(-1, 1))
                feature_names.append(col)
                transformations[col] = (
                    f"Binary encoding: {dict(zip(le.classes_, range(len(le.classes_))))}. "
                    f"Missing values: {missing_count} ({missing_pct:.1f}%) - PRESERVED AS NaN"
                )
            else:
                # One-hot encoding for multi-class - preserve NaN
                # Create dummy variables, NaN rows get 0 in all dummies
                dummies = pd.get_dummies(series, prefix=col, dummy_na=False)

                # Mark rows with NaN - these will be handled specially
                nan_mask = series.isna()
                dummies_array = dummies.values.astype(float)
                # Set NaN rows to NaN in all dummy columns
                dummies_array[nan_mask, :] = np.nan

                processed_features.append(dummies_array)
                feature_names.extend(dummies.columns.tolist())
                transformations[col] = (
                    f"One-hot encoding ({len(unique_vals)} categories). "
                    f"Missing values: {missing_count} ({missing_pct:.1f}%) - PRESERVED AS NaN"
                )
        else:
            # Numerical column - keep as-is with NaN preserved
            processed_features.append(series.values.reshape(-1, 1))
            feature_names.append(col)
            transformations[col] = (
                f"Numerical (kept as-is). "
                f"Missing values: {missing_count} ({missing_pct:.1f}%) - PRESERVED AS NaN"
            )

    # Concatenate all features
    X = np.hstack(processed_features)

    # Process target columns
    processed_targets = []
    target_names_list = []

    for col in target_columns:
        series = df_work[col]
        le = LabelEncoder()

        # For targets, we need to handle missing values
        missing_count = series.isna().sum()
        if missing_count > 0:
            # Encode non-NaN values, preserve NaN
            encoded = np.full(len(series), np.nan)
            non_nan_mask = ~series.isna()
            le.fit(series[non_nan_mask])
            encoded[non_nan_mask] = le.transform(series[non_nan_mask])
            transformations[f"TARGET_{col}"] = (
                f"Target encoding: {dict(zip(le.classes_, range(len(le.classes_))))}. "
                f"Missing values: {missing_count} - PRESERVED AS NaN")
        else:
            encoded = le.fit_transform(series)
            transformations[f"TARGET_{col}"] = (
                f"Target encoding: {dict(zip(le.classes_, range(len(le.classes_))))}"
            )

        processed_targets.append(encoded.reshape(-1, 1))
        target_names_list.append(col)

    y = np.hstack(processed_targets)

    return X, y, feature_names, target_names_list


# Process data with all features
print("\nProcessing data with all features...")
X_all, y_all, feature_names_all, target_names = preprocess_data(
    df_data, include_ultrasound=True)
print(f"Features shape: {X_all.shape}")
print(f"Targets shape: {y_all.shape}")

# Count missing values in features
total_missing = np.isnan(X_all).sum()
total_elements = X_all.size
print(
    f"Total missing values in features: {total_missing} ({100*total_missing/total_elements:.2f}%)"
)

# Process data without ultrasound features
print("\nProcessing data without ultrasound features...")
X_no_us, y_no_us, feature_names_no_us, _ = preprocess_data(
    df_data, include_ultrasound=False)
print(f"Features shape: {X_no_us.shape}")
print(f"Targets shape: {y_no_us.shape}")

total_missing_no_us = np.isnan(X_no_us).sum()
total_elements_no_us = X_no_us.size
print(
    f"Total missing values in features: {total_missing_no_us} ({100*total_missing_no_us/total_elements_no_us:.2f}%)"
)

# Split data into training and test sets FIRST
print("\n" + "=" * 60)
print("SPLITTING DATA INTO TRAIN AND TEST SETS")
print("=" * 60)

# Use the same indices for both versions to ensure consistency
# Stratify by Diagnosis (first target) to maintain class balance
test_size = 0.2
random_state = 42

# For stratification, we need to handle any NaN in the stratification column
# Get the Diagnosis column (index 0 in y_all)
y_diagnosis = y_all[:, 0]

# Check for NaN in Diagnosis
nan_diagnosis_mask = np.isnan(y_diagnosis)
if nan_diagnosis_mask.any():
    print(
        f"Warning: {nan_diagnosis_mask.sum()} samples have missing Diagnosis - removing them"
    )
    # Remove samples with missing Diagnosis
    valid_mask = ~nan_diagnosis_mask
    X_all = X_all[valid_mask]
    y_all = y_all[valid_mask]
    X_no_us = X_no_us[valid_mask]
    y_no_us = y_no_us[valid_mask]
    y_diagnosis = y_diagnosis[valid_mask]

# Split all features version
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all,
    y_all,
    test_size=test_size,
    random_state=random_state,
    stratify=y_diagnosis.astype(int))
print(f"\nAll Features:")
print(f"  Training set: {X_train_all.shape[0]} samples")
print(f"  Test set: {X_test_all.shape[0]} samples")

# Split no ultrasound version (use same indices via random state)
y_diagnosis_no_us = y_no_us[:, 0]
X_train_no_us, X_test_no_us, y_train_no_us, y_test_no_us = train_test_split(
    X_no_us,
    y_no_us,
    test_size=test_size,
    random_state=random_state,
    stratify=y_diagnosis_no_us.astype(int))
print(f"\nNo Ultrasound:")
print(f"  Training set: {X_train_no_us.shape[0]} samples")
print(f"  Test set: {X_test_no_us.shape[0]} samples")

# Save processed data (train and test splits)
print("\nSaving processed data...")

# All features - train
np.save('./processed_data/X_train_all_features.npy', X_train_all)
np.save('./processed_data/y_train_all_features.npy', y_train_all)
# All features - test
np.save('./processed_data/X_test_all_features.npy', X_test_all)
np.save('./processed_data/y_test_all_features.npy', y_test_all)

# No ultrasound - train
np.save('./processed_data/X_train_no_ultrasound.npy', X_train_no_us)
np.save('./processed_data/y_train_no_ultrasound.npy', y_train_no_us)
# No ultrasound - test
np.save('./processed_data/X_test_no_ultrasound.npy', X_test_no_us)
np.save('./processed_data/y_test_no_ultrasound.npy', y_test_no_us)

# Also save full data for analysis purposes
np.save('./processed_data/X_all_features.npy', X_all)
np.save('./processed_data/y_all_features.npy', y_all)
np.save('./processed_data/X_no_ultrasound.npy', X_no_us)
np.save('./processed_data/y_no_ultrasound.npy', y_no_us)

# Save feature names and target names
np.save('./processed_data/feature_names_all.npy',
        np.array(feature_names_all, dtype=object))
np.save('./processed_data/feature_names_no_ultrasound.npy',
        np.array(feature_names_no_us, dtype=object))
np.save('./processed_data/target_names.npy',
        np.array(target_names, dtype=object))

# Write transformation summary
print("\nWriting transformation summary...")
with open('./processed_data/transformation_summary.txt', 'w',
          encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("DATA PREPROCESSING TRANSFORMATION SUMMARY\n")
    f.write("Child Appendicitis Machine Learning Project\n")
    f.write("=" * 80 + "\n\n")

    f.write("OVERVIEW\n")
    f.write("-" * 40 + "\n")
    f.write(f"Original data shape: {df_data.shape}\n")
    f.write(f"Processed data with all features: {X_all.shape}\n")
    f.write(f"Processed data without ultrasound: {X_no_us.shape}\n")
    f.write(f"Target variables: {target_names}\n\n")

    f.write("TRAIN/TEST SPLIT\n")
    f.write("-" * 40 + "\n")
    f.write(f"Test size: {test_size} ({test_size*100:.0f}%)\n")
    f.write(f"Random state: {random_state}\n")
    f.write(f"Stratified by: Diagnosis (first target)\n")
    f.write(f"Training samples: {X_train_all.shape[0]}\n")
    f.write(f"Test samples: {X_test_all.shape[0]}\n\n")

    f.write("=" * 80 + "\n")
    f.write("MISSING VALUE HANDLING STRATEGY\n")
    f.write("=" * 80 + "\n")
    f.write("** MISSING VALUES ARE FAITHFULLY PRESERVED AS NaN **\n\n")
    f.write("This allows downstream models to:\n")
    f.write("  1. Use imputation strategies during training\n")
    f.write("  2. Apply different missing value handling methods\n")
    f.write("  3. Treat missingness as informative (if appropriate)\n\n")

    f.write("MISSING VALUE STATISTICS\n")
    f.write("-" * 40 + "\n")

    # Sort by missing percentage
    sorted_missing = sorted(missing_value_stats.items(),
                            key=lambda x: x[1]['percentage'],
                            reverse=True)

    for col, stats in sorted_missing:
        if stats['count'] > 0:
            f.write(
                f"{col}: {stats['count']} missing ({stats['percentage']:.1f}%)\n"
            )

    f.write("\n")
    f.write(f"Total missing in all features data: {total_missing} cells\n")
    f.write(
        f"Total missing in no-ultrasound data: {total_missing_no_us} cells\n\n"
    )

    f.write("FEATURE NAMES (All Features)\n")
    f.write("-" * 40 + "\n")
    for i, name in enumerate(feature_names_all):
        f.write(f"  {i+1}. {name}\n")
    f.write("\n")

    f.write("FEATURE NAMES (No Ultrasound)\n")
    f.write("-" * 40 + "\n")
    for i, name in enumerate(feature_names_no_us):
        f.write(f"  {i+1}. {name}\n")
    f.write("\n")

    f.write("COLUMN TRANSFORMATIONS\n")
    f.write("-" * 40 + "\n")
    for col, transform in sorted(transformations.items()):
        f.write(f"\n{col}:\n")
        f.write(f"  {transform}\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("ENCODING METHODS\n")
    f.write("-" * 40 + "\n")
    f.write(
        "- Binary categorical variables: Label encoding (0/1), NaN preserved\n"
    )
    f.write(
        "- Multi-class categorical variables: One-hot encoding, NaN preserved\n"
    )
    f.write(
        "- Numerical variables: Kept as continuous values, NaN preserved\n")
    f.write("- Target variables: Label encoding\n\n")

    f.write("ULTRASOUND COLUMNS (Excluded in second version)\n")
    f.write("-" * 40 + "\n")
    for col in ultrasound_columns:
        f.write(f"  - {col}\n")

print("\nData preprocessing completed successfully!")
print(f"Files saved in ./processed_data/")
print("\n** NOTE: Missing values are preserved as NaN in the data **")
print("** Models should handle missing values appropriately **")
