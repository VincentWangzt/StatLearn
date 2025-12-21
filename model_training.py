"""
Model Training Script for Child Appendicitis ML Project
Fits multiple models and evaluates their performance.
Handles missing values using imputation during training.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
import warnings
import os

warnings.filterwarnings('ignore')

# Ensure output directories exist
os.makedirs('./analysis', exist_ok=True)
os.makedirs('./models', exist_ok=True)  # Model results go here

# Load processed data
print("=" * 80)
print("CHILD APPENDICITIS PREDICTION - MODEL TRAINING")
print("=" * 80)

print("\nLoading processed data...")

# Load pre-split train and test data
X_train_all = np.load('./processed_data/X_train_all_features.npy',
                      allow_pickle=True)
X_test_all = np.load('./processed_data/X_test_all_features.npy',
                     allow_pickle=True)
y_train_all = np.load('./processed_data/y_train_all_features.npy',
                      allow_pickle=True)
y_test_all = np.load('./processed_data/y_test_all_features.npy',
                     allow_pickle=True)

X_train_no_us = np.load('./processed_data/X_train_no_ultrasound.npy',
                        allow_pickle=True)
X_test_no_us = np.load('./processed_data/X_test_no_ultrasound.npy',
                       allow_pickle=True)
y_train_no_us = np.load('./processed_data/y_train_no_ultrasound.npy',
                        allow_pickle=True)
y_test_no_us = np.load('./processed_data/y_test_no_ultrasound.npy',
                       allow_pickle=True)

# Load full data for feature selection
X_all = np.load('./processed_data/X_all_features.npy', allow_pickle=True)
y_all = np.load('./processed_data/y_all_features.npy', allow_pickle=True)
X_no_us = np.load('./processed_data/X_no_ultrasound.npy', allow_pickle=True)
y_no_us = np.load('./processed_data/y_no_ultrasound.npy', allow_pickle=True)

feature_names_all = np.load('./processed_data/feature_names_all.npy',
                            allow_pickle=True)
feature_names_no_us = np.load(
    './processed_data/feature_names_no_ultrasound.npy', allow_pickle=True)
target_names = np.load('./processed_data/target_names.npy', allow_pickle=True)

print(f"Training data (all features): {X_train_all.shape}")
print(f"Test data (all features): {X_test_all.shape}")
print(f"Training data (no ultrasound): {X_train_no_us.shape}")
print(f"Test data (no ultrasound): {X_test_no_us.shape}")
print(f"Targets: {target_names}")

# Report missing values
print(
    f"\nMissing values in training data (all features): {np.isnan(X_train_all).sum()}"
)
print(
    f"Missing values in test data (all features): {np.isnan(X_test_all).sum()}"
)
print(
    f"Missing values in training data (no ultrasound): {np.isnan(X_train_no_us).sum()}"
)
print(
    f"Missing values in test data (no ultrasound): {np.isnan(X_test_no_us).sum()}"
)


def get_models_with_imputation():
    """
    Return dictionary of models that can handle missing values.
    Uses pipelines with imputation for models that don't natively support NaN.
    """
    # Imputer for models that need complete data
    imputer = SimpleImputer(strategy='median')

    return {
        'Logistic Regression':
        Pipeline([('imputer', SimpleImputer(strategy='median')),
                  ('scaler', StandardScaler()),
                  ('clf', LogisticRegression(max_iter=1000,
                                             random_state=42))]),
        'Random Forest':
        Pipeline([('imputer', SimpleImputer(strategy='median')),
                  ('clf',
                   RandomForestClassifier(n_estimators=100,
                                          random_state=42,
                                          n_jobs=-1))]),
        'Gradient Boosting':
        Pipeline([('imputer', SimpleImputer(strategy='median')),
                  ('clf',
                   GradientBoostingClassifier(n_estimators=100,
                                              random_state=42))]),
        # HistGradientBoosting can handle NaN natively!
        'HistGradientBoosting':
        HistGradientBoostingClassifier(max_iter=100, random_state=42),
        'SVM':
        Pipeline([('imputer', SimpleImputer(strategy='median')),
                  ('scaler', StandardScaler()),
                  ('clf', SVC(kernel='rbf', probability=True,
                              random_state=42))]),
        'KNN':
        Pipeline([('imputer', SimpleImputer(strategy='median')),
                  ('scaler', StandardScaler()),
                  ('clf', KNeighborsClassifier(n_neighbors=5))]),
        'Decision Tree':
        Pipeline([('imputer', SimpleImputer(strategy='median')),
                  ('clf', DecisionTreeClassifier(random_state=42))])
    }


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model"""
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Handle multi-class metrics
    n_classes = len(np.unique(y_test))
    average = 'binary' if n_classes == 2 else 'weighted'

    precision = precision_score(y_test,
                                y_pred,
                                average=average,
                                zero_division=0)
    recall = recall_score(y_test, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=average, zero_division=0)

    # AUC-ROC
    auc = None
    if y_prob is not None:
        try:
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                auc = roc_auc_score(y_test,
                                    y_prob,
                                    multi_class='ovr',
                                    average='weighted')
        except:
            auc = None

    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def train_and_evaluate_all_models(X_train, X_test, y_train, y_test,
                                  target_name, feature_names, dataset_name):
    """Train and evaluate all models for a single target using pre-split data"""
    print(f"\n{'='*60}")
    print(f"Target: {target_name} | Dataset: {dataset_name}")
    print(f"{'='*60}")

    # Handle rare classes - remove samples with classes that have too few members
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    valid_classes = unique_train[
        counts_train
        >= 5]  # Only keep classes with at least 5 samples in training

    if len(valid_classes) < len(unique_train):
        print(
            f"  Warning: Removing rare classes with <5 samples in training set"
        )
        # Filter training data
        train_mask = np.isin(y_train, valid_classes)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        # Filter test data
        test_mask = np.isin(y_test, valid_classes)
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        # Re-encode labels to be consecutive
        label_map = {
            old: new
            for new, old in enumerate(sorted(np.unique(y_train)))
        }
        y_train = np.array([label_map[v] for v in y_train])
        y_test = np.array([label_map.get(v, -1) for v in y_test])
        # Remove any -1 (unknown classes in test)
        valid_test = y_test >= 0
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]
        print(
            f"  Training samples: {len(y_train)}, Test samples: {len(y_test)}")

    print(f"  Missing values in X_train: {np.isnan(X_train).sum()}")
    print(f"  Missing values in X_test: {np.isnan(X_test).sum()}")

    results = {}
    models = get_models_with_imputation()

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        result = evaluate_model(model, X_train, X_test, y_train, y_test,
                                model_name)
        results[model_name] = result

        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1_score']:.4f}")
        if result['auc_roc'] is not None:
            print(f"  AUC-ROC:   {result['auc_roc']:.4f}")

    return results, X_train, X_test, y_train, y_test


def feature_selection(X, y, feature_names, target_name, top_k=20):
    """Perform feature selection using multiple methods"""
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION for {target_name}")
    print(f"{'='*60}")

    # Handle rare classes
    unique, counts = np.unique(y, return_counts=True)
    valid_classes = unique[counts >= 5]
    if len(valid_classes) < len(unique):
        mask = np.isin(y, valid_classes)
        X = X[mask]
        y = y[mask]
        label_map = {old: new for new, old in enumerate(sorted(np.unique(y)))}
        y = np.array([label_map[v] for v in y])

    # For feature selection, we need to impute first
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Method 1: Random Forest Feature Importance
    print("\n1. Random Forest Feature Importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_scaled, y)
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"   Top {top_k} features:")
    for i, row in rf_importance.head(top_k).iterrows():
        print(f"   - {row['feature']}: {row['importance']:.4f}")

    # Method 2: Mutual Information
    print(f"\n2. Mutual Information...")
    mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
    mi_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mi_scores
    }).sort_values('importance', ascending=False)

    print(f"   Top {top_k} features:")
    for i, row in mi_importance.head(top_k).iterrows():
        print(f"   - {row['feature']}: {row['importance']:.4f}")

    # Method 3: L1 (Lasso) Regularization
    print(f"\n3. L1 Regularization (Logistic Regression)...")
    lr_l1 = LogisticRegression(penalty='l1',
                               solver='saga',
                               max_iter=1000,
                               random_state=42,
                               C=0.1)
    lr_l1.fit(X_scaled, y)

    # Get non-zero coefficients
    if len(lr_l1.coef_.shape) == 1:
        coef = np.abs(lr_l1.coef_)
    else:
        coef = np.mean(np.abs(lr_l1.coef_), axis=0)

    l1_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': coef
    }).sort_values('importance', ascending=False)

    print(f"   Top {top_k} features (non-zero):")
    l1_nonzero = l1_importance[l1_importance['importance'] > 0].head(top_k)
    for i, row in l1_nonzero.iterrows():
        print(f"   - {row['feature']}: {row['importance']:.4f}")

    # Get consensus top features
    top_rf = set(rf_importance.head(top_k)['feature'])
    top_mi = set(mi_importance.head(top_k)['feature'])
    top_l1 = set(l1_nonzero['feature'])

    consensus_features = list(top_rf & top_mi)  # Features in both RF and MI

    return {
        'rf_importance': rf_importance,
        'mi_importance': mi_importance,
        'l1_importance': l1_importance,
        'consensus_features': consensus_features,
        'rf_model': rf
    }


def train_with_selected_features(X_train, X_test, y_train, y_test,
                                 feature_names, selected_features,
                                 target_name):
    """Train models using only selected features with pre-split data"""
    print(f"\n{'='*60}")
    print(f"TRAINING WITH SELECTED FEATURES for {target_name}")
    print(f"Using {len(selected_features)} features")
    print(f"{'='*60}")

    # Handle rare classes
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    valid_classes = unique_train[counts_train >= 5]
    if len(valid_classes) < len(unique_train):
        train_mask = np.isin(y_train, valid_classes)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        test_mask = np.isin(y_test, valid_classes)
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        label_map = {
            old: new
            for new, old in enumerate(sorted(np.unique(y_train)))
        }
        y_train = np.array([label_map[v] for v in y_train])
        y_test = np.array([label_map.get(v, -1) for v in y_test])
        valid_test = y_test >= 0
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]

    # Get indices of selected features
    feature_idx = [
        i for i, f in enumerate(feature_names) if f in selected_features
    ]
    X_train_selected = X_train[:, feature_idx]
    X_test_selected = X_test[:, feature_idx]

    results = {}
    models = get_models_with_imputation()

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        result = evaluate_model(model, X_train_selected, X_test_selected,
                                y_train, y_test, model_name)
        results[model_name] = result

        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  F1-Score:  {result['f1_score']:.4f}")
        if result['auc_roc'] is not None:
            print(f"  AUC-ROC:   {result['auc_roc']:.4f}")

    return results


def create_results_summary_table(all_results):
    """Create a summary table of all results"""
    rows = []
    for (target, dataset), results in all_results.items():
        for model_name, metrics in results.items():
            rows.append({
                'Target':
                target,
                'Dataset':
                dataset,
                'Model':
                model_name,
                'Accuracy':
                metrics['accuracy'],
                'Precision':
                metrics['precision'],
                'Recall':
                metrics['recall'],
                'F1-Score':
                metrics['f1_score'],
                'AUC-ROC':
                metrics['auc_roc'] if metrics['auc_roc'] else np.nan
            })

    df = pd.DataFrame(rows)
    return df


def plot_model_comparison(results_df, save_path):
    """Create comparison plots for all models"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        pivot_df = results_df.pivot_table(values=metric,
                                          index='Model',
                                          columns=['Target', 'Dataset'],
                                          aggfunc='mean')

        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(f'{metric} Comparison', fontsize=12)
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.legend(title='Target/Dataset',
                  bbox_to_anchor=(1.05, 1),
                  loc='upper left')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(results, y_test, target_name, dataset_name,
                            save_path):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    # Hide extra subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f'Confusion Matrices - {target_name} ({dataset_name})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(feature_selection_results, target_name, save_path):
    """Plot feature importance from different methods"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # RF Importance
    ax = axes[0]
    top_rf = feature_selection_results['rf_importance'].head(15)
    ax.barh(range(len(top_rf)), top_rf['importance'], color='steelblue')
    ax.set_yticks(range(len(top_rf)))
    ax.set_yticklabels(top_rf['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest Feature Importance')

    # MI Importance
    ax = axes[1]
    top_mi = feature_selection_results['mi_importance'].head(15)
    ax.barh(range(len(top_mi)), top_mi['importance'], color='forestgreen')
    ax.set_yticks(range(len(top_mi)))
    ax.set_yticklabels(top_mi['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('Mutual Information')

    # L1 Importance
    ax = axes[2]
    top_l1 = feature_selection_results['l1_importance'].head(15)
    ax.barh(range(len(top_l1)), top_l1['importance'], color='coral')
    ax.set_yticks(range(len(top_l1)))
    ax.set_yticklabels(top_l1['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title('L1 Regularization')

    plt.suptitle(f'Feature Importance - {target_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# Main training loop
print("\n" + "=" * 80)
print("STARTING MODEL TRAINING (Using Pre-Split Train/Test Data)")
print("Missing values are handled via imputation during training")
print("=" * 80)

all_results = {}
feature_selection_all = {}

# Train for each target
for target_idx, target_name in enumerate(target_names):
    # Get pre-split data for this target
    y_train = y_train_all[:, target_idx]
    y_test = y_test_all[:, target_idx]

    # Train with all features using pre-split data
    results_all, X_tr, X_te, y_tr, y_te = train_and_evaluate_all_models(
        X_train_all.copy(), X_test_all.copy(), y_train.copy(), y_test.copy(),
        target_name, feature_names_all, "All Features")
    all_results[(target_name, 'All Features')] = results_all

    # Plot confusion matrices
    plot_confusion_matrices(
        results_all, y_te, target_name, "All Features",
        f'./analysis/confusion_matrix_{target_name}_all.png')

    # Train without ultrasound features using pre-split data
    y_train_no = y_train_no_us[:, target_idx]
    y_test_no = y_test_no_us[:, target_idx]
    results_no_us, _, _, _, _ = train_and_evaluate_all_models(
        X_train_no_us.copy(), X_test_no_us.copy(), y_train_no.copy(),
        y_test_no.copy(), target_name, feature_names_no_us, "No Ultrasound")
    all_results[(target_name, 'No Ultrasound')] = results_no_us

    # Feature selection (using full training data)
    y_full = y_all[:, target_idx]
    fs_results = feature_selection(X_all, y_full, feature_names_all,
                                   target_name)
    feature_selection_all[target_name] = fs_results

    # Plot feature importance
    plot_feature_importance(
        fs_results, target_name,
        f'./analysis/feature_importance_{target_name}.png')

    # Train with selected features using pre-split data
    selected_features = fs_results['rf_importance'].head(
        20)['feature'].tolist()
    results_selected = train_with_selected_features(X_train_all.copy(),
                                                    X_test_all.copy(),
                                                    y_train.copy(),
                                                    y_test.copy(),
                                                    feature_names_all,
                                                    selected_features,
                                                    target_name)
    all_results[(target_name, 'Selected Features')] = results_selected

# Create summary table
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

results_df = create_results_summary_table(all_results)
print("\n" + results_df.to_string())

# Save results to ./models/ directory
results_df.to_csv('./models/model_results.csv', index=False)

# Plot comparisons
plot_model_comparison(results_df, './analysis/model_comparison.png')

# Find best models for each target
print("\n" + "=" * 80)
print("BEST MODELS FOR EACH TARGET")
print("=" * 80)

for target in target_names:
    target_results = results_df[results_df['Target'] == target]
    best_row = target_results.loc[target_results['F1-Score'].idxmax()]
    print(f"\n{target}:")
    print(f"  Best Model: {best_row['Model']} ({best_row['Dataset']})")
    print(f"  Accuracy: {best_row['Accuracy']:.4f}")
    print(f"  F1-Score: {best_row['F1-Score']:.4f}")
    if pd.notna(best_row['AUC-ROC']):
        print(f"  AUC-ROC:  {best_row['AUC-ROC']:.4f}")

# Save feature selection results to ./models/ directory
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

for target_name, fs_results in feature_selection_all.items():
    fs_results['rf_importance'].to_csv(
        f'./models/feature_importance_rf_{target_name}.csv', index=False)
    fs_results['mi_importance'].to_csv(
        f'./models/feature_importance_mi_{target_name}.csv', index=False)

print("\nAll results saved!")
print(f"  - Model results: ./models/model_results.csv")
print(f"  - Feature importance: ./models/feature_importance_*.csv")
print(f"  - Plots: ./analysis/")

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED")
print("=" * 80)
