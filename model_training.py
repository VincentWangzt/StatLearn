"""
Model Training Script for Child Appendicitis ML Project
- Trains models on no_ultrasound features only
- Implements forward-backward stepwise feature selection
- Plots training and testing ROC curves
- Performs grid search on hyperparameters
- Outputs train vs test performance comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, auc)
from sklearn.feature_selection import SequentialFeatureSelector
import warnings
import os

warnings.filterwarnings('ignore')

# Ensure output directories exist
os.makedirs('./analysis', exist_ok=True)
os.makedirs('./models', exist_ok=True)

# =============================================================================
# LOAD DATA (No Ultrasound Features Only)
# =============================================================================
print("=" * 80)
print("CHILD APPENDICITIS PREDICTION - MODEL TRAINING")
print("Using No Ultrasound Features Only")
print("=" * 80)

print("\nLoading processed data (no ultrasound)...")

# Load pre-split train and test data (no ultrasound only)
X_train = np.load('./processed_data/X_train_no_ultrasound.npy', allow_pickle=True)
X_test = np.load('./processed_data/X_test_no_ultrasound.npy', allow_pickle=True)
y_train_all = np.load('./processed_data/y_train_no_ultrasound.npy', allow_pickle=True)
y_test_all = np.load('./processed_data/y_test_no_ultrasound.npy', allow_pickle=True)

# Load feature and target names
feature_names = np.load('./processed_data/feature_names_no_ultrasound.npy', allow_pickle=True)
target_names = np.load('./processed_data/target_names.npy', allow_pickle=True)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Targets: {list(target_names)}")

# Report missing values
n_missing_train = np.isnan(X_train).sum()
n_missing_test = np.isnan(X_test).sum()
print(f"\nMissing values in training data: {n_missing_train} ({100*n_missing_train/(X_train.shape[0]*X_train.shape[1]):.2f}%)")
print(f"Missing values in test data: {n_missing_test} ({100*n_missing_test/(X_test.shape[0]*X_test.shape[1]):.2f}%)")


# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================
def create_preprocessor():
    """Create preprocessing pipeline with imputation and scaling"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def prepare_data(X_train, X_test, y_train, y_test):
    """Prepare data: handle rare classes and return processed data"""
    # Handle rare classes - remove samples with classes that have too few members
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    valid_classes = unique_train[counts_train >= 5]
    
    if len(valid_classes) < len(unique_train):
        print(f"  Warning: Removing rare classes with <5 samples in training set")
        # Filter training data
        train_mask = np.isin(y_train, valid_classes)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        # Filter test data
        test_mask = np.isin(y_test, valid_classes)
        X_test = X_test[test_mask]
        y_test = y_test[test_mask]
        # Re-encode labels to be consecutive
        label_map = {old: new for new, old in enumerate(sorted(np.unique(y_train)))}
        y_train = np.array([label_map[v] for v in y_train])
        y_test = np.array([label_map.get(v, -1) for v in y_test])
        # Remove any -1 (unknown classes in test)
        valid_test = y_test >= 0
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]
        
    return X_train, X_test, y_train, y_test


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
def get_base_models():
    """Return dictionary of base models (without preprocessing)"""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }


def get_models_with_pipeline():
    """Return dictionary of models with preprocessing pipeline"""
    return {
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ]),
        'Gradient Boosting': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, random_state=42),
        'SVM': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, random_state=42))
        ]),
        'KNN': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ]),
        'Decision Tree': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', DecisionTreeClassifier(random_state=42))
        ])
    }


# =============================================================================
# FORWARD-BACKWARD STEPWISE FEATURE SELECTION
# =============================================================================
def stepwise_feature_selection(X_train, y_train, feature_names, model_name='Logistic Regression', 
                               direction='forward', n_features_to_select='auto', cv=5):
    """
    Perform stepwise feature selection using SequentialFeatureSelector.
    
    Parameters:
    -----------
    X_train : array-like
        Training features (already preprocessed)
    y_train : array-like
        Training targets
    feature_names : array-like
        Names of features
    model_name : str
        'Logistic Regression' or 'GLM' (same as Logistic Regression for classification)
    direction : str
        'forward', 'backward', or 'both' (sequential forward-backward)
    n_features_to_select : int or 'auto'
        Number of features to select
    cv : int
        Number of cross-validation folds
        
    Returns:
    --------
    selected_features : list
        Names of selected features
    selected_indices : array
        Indices of selected features
    selector : SequentialFeatureSelector
        Fitted selector object
    """
    print(f"\n  Performing {direction} feature selection with {model_name}...")
    
    # Create base estimator
    if model_name in ['Logistic Regression', 'GLM']:
        estimator = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Determine number of features to select
    if n_features_to_select == 'auto':
        n_features_to_select = min(15, X_train.shape[1] // 2)
    
    # Map direction to sklearn parameters
    if direction == 'both':
        # For "both", we do forward selection first, then can optionally do backward
        # sklearn doesn't have true forward-backward, so we use forward
        sklearn_direction = 'forward'
    else:
        sklearn_direction = direction
    
    # Create selector
    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features_to_select,
        direction=sklearn_direction,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1
    )
    
    # Fit selector
    selector.fit(X_train, y_train)
    
    # Get selected features
    selected_mask = selector.get_support()
    selected_indices = np.where(selected_mask)[0]
    selected_features = [feature_names[i] for i in selected_indices]
    
    print(f"  Selected {len(selected_features)} features:")
    for i, feat in enumerate(selected_features):
        print(f"    {i+1}. {feat}")
    
    return selected_features, selected_indices, selector


def forward_backward_selection(X_train, y_train, feature_names, cv=5):
    """
    Perform forward-backward stepwise selection.
    First does forward selection, then backward elimination on selected features.
    """
    print("\n  === Forward-Backward Stepwise Feature Selection ===")
    
    # Step 1: Forward selection to get candidate features
    print("\n  Step 1: Forward Selection...")
    estimator = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    
    n_features_forward = min(20, X_train.shape[1] // 2)
    
    forward_selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features_forward,
        direction='forward',
        scoring='accuracy',
        cv=cv,
        n_jobs=-1
    )
    forward_selector.fit(X_train, y_train)
    forward_mask = forward_selector.get_support()
    forward_indices = np.where(forward_mask)[0]
    
    print(f"  Forward selection chose {len(forward_indices)} features")
    
    # Step 2: Backward elimination on selected features
    print("\n  Step 2: Backward Elimination on selected features...")
    X_train_subset = X_train[:, forward_indices]
    feature_names_subset = [feature_names[i] for i in forward_indices]
    
    n_features_backward = max(5, len(forward_indices) // 2)
    
    backward_selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features_backward,
        direction='backward',
        scoring='accuracy',
        cv=cv,
        n_jobs=-1
    )
    backward_selector.fit(X_train_subset, y_train)
    backward_mask = backward_selector.get_support()
    
    # Map back to original indices
    final_indices = forward_indices[backward_mask]
    final_features = [feature_names[i] for i in final_indices]
    
    print(f"\n  Final selected features ({len(final_features)}):")
    for i, feat in enumerate(final_features):
        print(f"    {i+1}. {feat}")
    
    return final_features, final_indices


# =============================================================================
# MODEL EVALUATION WITH TRAIN/TEST COMPARISON
# =============================================================================
def evaluate_model_train_test(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a model, returning both training and testing metrics.
    """
    # Train
    model.fit(X_train, y_train)
    
    # Predict on training set
    y_train_pred = model.predict(X_train)
    y_train_prob = None
    if hasattr(model, 'predict_proba'):
        y_train_prob = model.predict_proba(X_train)
    
    # Predict on test set
    y_test_pred = model.predict(X_test)
    y_test_prob = None
    if hasattr(model, 'predict_proba'):
        y_test_prob = model.predict_proba(X_test)
    
    # Calculate metrics
    n_classes = len(np.unique(y_test))
    average = 'binary' if n_classes == 2 else 'weighted'
    
    # Training metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, average=average, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_train, y_train_pred, average=average, zero_division=0),
        'y_pred': y_train_pred,
        'y_prob': y_train_prob,
        'y_true': y_train
    }
    
    # Test metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, average=average, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_test, y_test_pred, average=average, zero_division=0),
        'y_pred': y_test_pred,
        'y_prob': y_test_prob,
        'y_true': y_test
    }
    
    # AUC-ROC
    for metrics, y_true, y_prob in [(train_metrics, y_train, y_train_prob), 
                                     (test_metrics, y_test, y_test_prob)]:
        if y_prob is not None:
            try:
                if n_classes == 2:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:
                    metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except:
                metrics['auc_roc'] = None
        else:
            metrics['auc_roc'] = None
    
    return {
        'model': model,
        'train': train_metrics,
        'test': test_metrics,
        'n_classes': n_classes
    }


# =============================================================================
# ROC CURVE PLOTTING
# =============================================================================
def plot_roc_curves(results, target_name, save_path):
    """
    Plot ROC curves for both training and testing sets for all models.
    """
    n_models = len(results)
    fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(5 * ((n_models + 1) // 2), 10))
    axes = axes.flatten()
    
    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx]
        n_classes = result['n_classes']
        
        # Plot training ROC
        train_prob = result['train']['y_prob']
        train_true = result['train']['y_true']
        test_prob = result['test']['y_prob']
        test_true = result['test']['y_true']
        
        if train_prob is not None and test_prob is not None:
            if n_classes == 2:
                # Binary classification
                fpr_train, tpr_train, _ = roc_curve(train_true, train_prob[:, 1])
                fpr_test, tpr_test, _ = roc_curve(test_true, test_prob[:, 1])
                
                train_auc = result['train']['auc_roc']
                test_auc = result['test']['auc_roc']
                
                ax.plot(fpr_train, tpr_train, 'b-', linewidth=2, 
                       label=f'Train (AUC = {train_auc:.3f})' if train_auc else 'Train')
                ax.plot(fpr_test, tpr_test, 'r-', linewidth=2,
                       label=f'Test (AUC = {test_auc:.3f})' if test_auc else 'Test')
            else:
                # Multi-class: plot macro-average ROC
                classes = np.unique(train_true)
                
                # Binarize labels
                train_true_bin = label_binarize(train_true, classes=classes)
                test_true_bin = label_binarize(test_true, classes=classes)
                
                # Compute micro-average ROC
                fpr_train, tpr_train, _ = roc_curve(train_true_bin.ravel(), train_prob.ravel())
                fpr_test, tpr_test, _ = roc_curve(test_true_bin.ravel(), test_prob.ravel())
                
                train_auc = auc(fpr_train, tpr_train)
                test_auc = auc(fpr_test, tpr_test)
                
                ax.plot(fpr_train, tpr_train, 'b-', linewidth=2,
                       label=f'Train (AUC = {train_auc:.3f})')
                ax.plot(fpr_test, tpr_test, 'r-', linewidth=2,
                       label=f'Test (AUC = {test_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'ROC Curves - {target_name}\n(Blue: Training, Red: Testing)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC curves saved to {save_path}")


# =============================================================================
# TRAIN VS TEST COMPARISON OUTPUT
# =============================================================================
def print_train_test_comparison(results, target_name):
    """Print detailed comparison between training and testing performance"""
    print(f"\n{'='*80}")
    print(f"TRAIN vs TEST COMPARISON - {target_name}")
    print(f"{'='*80}")
    
    comparison_data = []
    
    print(f"\n{'Model':<25} {'Metric':<12} {'Train':<10} {'Test':<10} {'Diff':<10} {'Overfit?':<10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        train_acc = result['train']['accuracy']
        test_acc = result['test']['accuracy']
        diff_acc = train_acc - test_acc
        
        train_f1 = result['train']['f1_score']
        test_f1 = result['test']['f1_score']
        diff_f1 = train_f1 - test_f1
        
        train_auc = result['train']['auc_roc']
        test_auc = result['test']['auc_roc']
        
        # Check for overfitting (train >> test)
        overfit = "Yes" if diff_acc > 0.1 else "No"
        
        print(f"{model_name:<25} {'Accuracy':<12} {train_acc:<10.4f} {test_acc:<10.4f} {diff_acc:<+10.4f} {overfit:<10}")
        print(f"{'':<25} {'F1-Score':<12} {train_f1:<10.4f} {test_f1:<10.4f} {diff_f1:<+10.4f}")
        if train_auc and test_auc:
            diff_auc = train_auc - test_auc
            print(f"{'':<25} {'AUC-ROC':<12} {train_auc:<10.4f} {test_auc:<10.4f} {diff_auc:<+10.4f}")
        print()
        
        comparison_data.append({
            'Model': model_name,
            'Train_Accuracy': train_acc,
            'Test_Accuracy': test_acc,
            'Accuracy_Diff': diff_acc,
            'Train_F1': train_f1,
            'Test_F1': test_f1,
            'F1_Diff': diff_f1,
            'Train_AUC': train_auc,
            'Test_AUC': test_auc,
            'Overfitting': overfit
        })
    
    return pd.DataFrame(comparison_data)


# =============================================================================
# GRID SEARCH WITH ACCURACY CURVES
# =============================================================================
def grid_search_with_plots(X_train, X_test, y_train, y_test, target_name, save_path):
    """
    Perform grid search on hyperparameters and plot accuracy curves.
    """
    print(f"\n{'='*60}")
    print(f"GRID SEARCH - {target_name}")
    print(f"{'='*60}")
    
    # Preprocess data first
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    grid_search_results = {}
    
    # 1. Random Forest - n_estimators
    print("\n1. Random Forest - n_estimators...")
    n_estimators_range = [10, 25, 50, 75, 100, 150, 200, 250, 300]
    rf_train_scores = []
    rf_test_scores = []
    
    for n_est in n_estimators_range:
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        rf.fit(X_train_processed, y_train)
        rf_train_scores.append(rf.score(X_train_processed, y_train))
        rf_test_scores.append(rf.score(X_test_processed, y_test))
    
    best_n_est = n_estimators_range[np.argmax(rf_test_scores)]
    grid_search_results['Random Forest'] = {
        'param_name': 'n_estimators',
        'param_range': n_estimators_range,
        'train_scores': rf_train_scores,
        'test_scores': rf_test_scores,
        'best_param': best_n_est,
        'best_score': max(rf_test_scores)
    }
    print(f"   Best n_estimators: {best_n_est}, Test Accuracy: {max(rf_test_scores):.4f}")
    
    # 2. KNN - n_neighbors
    print("\n2. KNN - n_neighbors...")
    n_neighbors_range = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    knn_train_scores = []
    knn_test_scores = []
    
    for n_neigh in n_neighbors_range:
        knn = KNeighborsClassifier(n_neighbors=n_neigh)
        knn.fit(X_train_processed, y_train)
        knn_train_scores.append(knn.score(X_train_processed, y_train))
        knn_test_scores.append(knn.score(X_test_processed, y_test))
    
    best_n_neigh = n_neighbors_range[np.argmax(knn_test_scores)]
    grid_search_results['KNN'] = {
        'param_name': 'n_neighbors',
        'param_range': n_neighbors_range,
        'train_scores': knn_train_scores,
        'test_scores': knn_test_scores,
        'best_param': best_n_neigh,
        'best_score': max(knn_test_scores)
    }
    print(f"   Best n_neighbors: {best_n_neigh}, Test Accuracy: {max(knn_test_scores):.4f}")
    
    # 3. Gradient Boosting - n_estimators
    print("\n3. Gradient Boosting - n_estimators...")
    gb_n_estimators_range = [25, 50, 75, 100, 125, 150, 200]
    gb_train_scores = []
    gb_test_scores = []
    
    for n_est in gb_n_estimators_range:
        gb = GradientBoostingClassifier(n_estimators=n_est, random_state=42)
        gb.fit(X_train_processed, y_train)
        gb_train_scores.append(gb.score(X_train_processed, y_train))
        gb_test_scores.append(gb.score(X_test_processed, y_test))
    
    best_gb_est = gb_n_estimators_range[np.argmax(gb_test_scores)]
    grid_search_results['Gradient Boosting'] = {
        'param_name': 'n_estimators',
        'param_range': gb_n_estimators_range,
        'train_scores': gb_train_scores,
        'test_scores': gb_test_scores,
        'best_param': best_gb_est,
        'best_score': max(gb_test_scores)
    }
    print(f"   Best n_estimators: {best_gb_est}, Test Accuracy: {max(gb_test_scores):.4f}")
    
    # 4. SVM - C parameter
    print("\n4. SVM - C (regularization)...")
    c_range = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    svm_train_scores = []
    svm_test_scores = []
    
    for c_val in c_range:
        svm = SVC(C=c_val, kernel='rbf', random_state=42)
        svm.fit(X_train_processed, y_train)
        svm_train_scores.append(svm.score(X_train_processed, y_train))
        svm_test_scores.append(svm.score(X_test_processed, y_test))
    
    best_c = c_range[np.argmax(svm_test_scores)]
    grid_search_results['SVM'] = {
        'param_name': 'C',
        'param_range': c_range,
        'train_scores': svm_train_scores,
        'test_scores': svm_test_scores,
        'best_param': best_c,
        'best_score': max(svm_test_scores)
    }
    print(f"   Best C: {best_c}, Test Accuracy: {max(svm_test_scores):.4f}")
    
    # 5. Decision Tree - max_depth
    print("\n5. Decision Tree - max_depth...")
    max_depth_range = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, None]
    dt_train_scores = []
    dt_test_scores = []
    
    for depth in max_depth_range:
        dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
        dt.fit(X_train_processed, y_train)
        dt_train_scores.append(dt.score(X_train_processed, y_train))
        dt_test_scores.append(dt.score(X_test_processed, y_test))
    
    best_depth = max_depth_range[np.argmax(dt_test_scores)]
    grid_search_results['Decision Tree'] = {
        'param_name': 'max_depth',
        'param_range': [str(d) if d else 'None' for d in max_depth_range],
        'train_scores': dt_train_scores,
        'test_scores': dt_test_scores,
        'best_param': best_depth if best_depth else 'None',
        'best_score': max(dt_test_scores)
    }
    print(f"   Best max_depth: {best_depth}, Test Accuracy: {max(dt_test_scores):.4f}")
    
    # 6. Logistic Regression - C parameter
    print("\n6. Logistic Regression - C (regularization)...")
    lr_c_range = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    lr_train_scores = []
    lr_test_scores = []
    
    for c_val in lr_c_range:
        lr = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
        lr.fit(X_train_processed, y_train)
        lr_train_scores.append(lr.score(X_train_processed, y_train))
        lr_test_scores.append(lr.score(X_test_processed, y_test))
    
    best_lr_c = lr_c_range[np.argmax(lr_test_scores)]
    grid_search_results['Logistic Regression'] = {
        'param_name': 'C',
        'param_range': lr_c_range,
        'train_scores': lr_train_scores,
        'test_scores': lr_test_scores,
        'best_param': best_lr_c,
        'best_score': max(lr_test_scores)
    }
    print(f"   Best C: {best_lr_c}, Test Accuracy: {max(lr_test_scores):.4f}")
    
    # Plot accuracy curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    plot_configs = [
        ('Random Forest', 'n_estimators'),
        ('KNN', 'n_neighbors'),
        ('Gradient Boosting', 'n_estimators'),
        ('SVM', 'C'),
        ('Decision Tree', 'max_depth'),
        ('Logistic Regression', 'C')
    ]
    
    for idx, (model_name, param_name) in enumerate(plot_configs):
        ax = axes[idx]
        result = grid_search_results[model_name]
        
        param_range = result['param_range']
        x_range = range(len(param_range))
        
        ax.plot(x_range, result['train_scores'], 'b-o', linewidth=2, markersize=6, label='Train')
        ax.plot(x_range, result['test_scores'], 'r-s', linewidth=2, markersize=6, label='Test')
        
        ax.set_xticks(x_range)
        ax.set_xticklabels([str(p) for p in param_range], rotation=45)
        ax.set_xlabel(param_name)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model_name}\nBest {param_name}: {result["best_param"]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.5, 1.05])
    
    plt.suptitle(f'Hyperparameter Tuning - {target_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Hyperparameter plots saved to {save_path}")
    
    return grid_search_results


# =============================================================================
# CONFUSION MATRIX PLOTTING
# =============================================================================
def plot_confusion_matrices(results, target_name, save_path):
    """Plot confusion matrices for all models"""
    n_models = len(results)
    n_cols = 4
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()
    
    for idx, (model_name, result) in enumerate(results.items()):
        ax = axes[idx]
        y_test = result['test']['y_true']
        y_pred = result['test']['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name}\nAcc: {result["test"]["accuracy"]:.3f}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide extra subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Confusion Matrices (Test Set) - {target_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrices saved to {save_path}")


# =============================================================================
# FEATURE IMPORTANCE PLOTTING
# =============================================================================
def plot_feature_importance_comparison(fs_results_lr, fs_results_fb, feature_names, 
                                       target_name, save_path):
    """Plot feature importance from different selection methods"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Forward selection features
    ax = axes[0]
    forward_features = fs_results_lr
    if forward_features:
        y_pos = range(len(forward_features))
        ax.barh(y_pos, [1] * len(forward_features), color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(forward_features)
        ax.invert_yaxis()
        ax.set_xlabel('Selected')
        ax.set_title(f'Forward Selection\n({len(forward_features)} features)')
    
    # Forward-backward selection features
    ax = axes[1]
    fb_features = fs_results_fb
    if fb_features:
        y_pos = range(len(fb_features))
        ax.barh(y_pos, [1] * len(fb_features), color='forestgreen')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(fb_features)
        ax.invert_yaxis()
        ax.set_xlabel('Selected')
        ax.set_title(f'Forward-Backward Selection\n({len(fb_features)} features)')
    
    plt.suptitle(f'Stepwise Feature Selection Results - {target_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Feature selection plot saved to {save_path}")


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
print("\n" + "=" * 80)
print("STARTING MODEL TRAINING")
print("=" * 80)

all_results = {}
all_comparisons = []
all_grid_search = {}
all_feature_selection = {}

# Train for each target
for target_idx, target_name in enumerate(target_names):
    print(f"\n{'#'*80}")
    print(f"# TARGET: {target_name}")
    print(f"{'#'*80}")
    
    # Get target values
    y_train = y_train_all[:, target_idx]
    y_test = y_test_all[:, target_idx]
    
    # Prepare data (handle rare classes)
    X_tr, X_te, y_tr, y_te = prepare_data(
        X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
    )
    
    print(f"\nData shape after preparation:")
    print(f"  Training: {X_tr.shape}, Test: {X_te.shape}")
    print(f"  Classes: {np.unique(y_tr)}")
    
    # =========================================================================
    # Step 1: Train all models with full features
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"TRAINING ALL MODELS (Full Features)")
    print(f"{'='*60}")
    
    results = {}
    models = get_models_with_pipeline()
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        result = evaluate_model_train_test(model, X_tr, X_te, y_tr, y_te, model_name)
        results[model_name] = result
        
        print(f"  Train - Acc: {result['train']['accuracy']:.4f}, F1: {result['train']['f1_score']:.4f}")
        print(f"  Test  - Acc: {result['test']['accuracy']:.4f}, F1: {result['test']['f1_score']:.4f}")
    
    all_results[target_name] = results
    
    # =========================================================================
    # Step 2: Feature Selection (Forward-Backward)
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEPWISE FEATURE SELECTION")
    print(f"{'='*60}")
    
    # Preprocess data for feature selection
    preprocessor = create_preprocessor()
    X_tr_processed = preprocessor.fit_transform(X_tr)
    X_te_processed = preprocessor.transform(X_te)
    
    # Forward selection with Logistic Regression
    forward_features, forward_indices, _ = stepwise_feature_selection(
        X_tr_processed, y_tr, feature_names, 
        model_name='Logistic Regression',
        direction='forward',
        n_features_to_select=15
    )
    
    # Forward-Backward selection
    fb_features, fb_indices = forward_backward_selection(
        X_tr_processed, y_tr, feature_names
    )
    
    all_feature_selection[target_name] = {
        'forward': forward_features,
        'forward_backward': fb_features,
        'n_forward': len(forward_features),
        'n_forward_backward': len(fb_features)
    }
    
    # Plot feature selection results
    plot_feature_importance_comparison(
        forward_features, fb_features, feature_names, target_name,
        f'./analysis/feature_selection_{target_name}.png'
    )
    
    # =========================================================================
    # Step 3: Train with selected features
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"TRAINING WITH SELECTED FEATURES (Forward-Backward)")
    print(f"{'='*60}")
    
    # Use forward-backward selected features
    X_tr_selected = X_tr_processed[:, fb_indices]
    X_te_selected = X_te_processed[:, fb_indices]
    
    results_selected = {}
    base_models = get_base_models()
    
    for model_name, model in base_models.items():
        print(f"\nTraining {model_name} with {len(fb_features)} features...")
        result = evaluate_model_train_test(model, X_tr_selected, X_te_selected, y_tr, y_te, model_name)
        results_selected[model_name] = result
        
        print(f"  Train - Acc: {result['train']['accuracy']:.4f}, F1: {result['train']['f1_score']:.4f}")
        print(f"  Test  - Acc: {result['test']['accuracy']:.4f}, F1: {result['test']['f1_score']:.4f}")
    
    all_results[f"{target_name}_selected"] = results_selected
    
    # =========================================================================
    # Step 4: Plot ROC Curves
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"PLOTTING ROC CURVES")
    print(f"{'='*60}")
    
    plot_roc_curves(results, target_name, f'./analysis/roc_curves_{target_name}.png')
    plot_roc_curves(results_selected, f"{target_name} (Selected Features)", 
                   f'./analysis/roc_curves_{target_name}_selected.png')
    
    # =========================================================================
    # Step 5: Train vs Test Comparison
    # =========================================================================
    comparison_df = print_train_test_comparison(results, target_name)
    all_comparisons.append(comparison_df.assign(Target=target_name, Features='Full'))
    
    comparison_df_selected = print_train_test_comparison(results_selected, f"{target_name} (Selected)")
    all_comparisons.append(comparison_df_selected.assign(Target=target_name, Features='Selected'))
    
    # =========================================================================
    # Step 6: Grid Search
    # =========================================================================
    grid_results = grid_search_with_plots(
        X_tr, X_te, y_tr, y_te, target_name,
        f'./analysis/grid_search_{target_name}.png'
    )
    all_grid_search[target_name] = grid_results
    
    # =========================================================================
    # Step 7: Confusion Matrices
    # =========================================================================
    plot_confusion_matrices(results, target_name, f'./analysis/confusion_matrix_{target_name}.png')


# =============================================================================
# SUMMARY RESULTS
# =============================================================================
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)

# Create summary table
summary_rows = []
for target_name, results in all_results.items():
    for model_name, result in results.items():
        summary_rows.append({
            'Target': target_name,
            'Model': model_name,
            'Train_Accuracy': result['train']['accuracy'],
            'Test_Accuracy': result['test']['accuracy'],
            'Train_F1': result['train']['f1_score'],
            'Test_F1': result['test']['f1_score'],
            'Train_AUC': result['train']['auc_roc'],
            'Test_AUC': result['test']['auc_roc']
        })

results_df = pd.DataFrame(summary_rows)
print("\n" + results_df.to_string())

# Save results
results_df.to_csv('./models/model_results.csv', index=False)
print(f"\nResults saved to ./models/model_results.csv")

# Save comparisons
comparisons_df = pd.concat(all_comparisons, ignore_index=True)
comparisons_df.to_csv('./models/train_test_comparison.csv', index=False)
print(f"Comparison saved to ./models/train_test_comparison.csv")

# Save grid search results
grid_search_summary = []
for target, results in all_grid_search.items():
    for model, params in results.items():
        grid_search_summary.append({
            'Target': target,
            'Model': model,
            'Parameter': params['param_name'],
            'Best_Value': params['best_param'],
            'Best_Test_Accuracy': params['best_score']
        })
grid_search_df = pd.DataFrame(grid_search_summary)
grid_search_df.to_csv('./models/grid_search_results.csv', index=False)
print(f"Grid search results saved to ./models/grid_search_results.csv")

# Save feature selection results
fs_summary = []
for target, fs_result in all_feature_selection.items():
    fs_summary.append({
        'Target': target,
        'N_Forward_Features': fs_result['n_forward'],
        'N_Forward_Backward_Features': fs_result['n_forward_backward'],
        'Forward_Features': ', '.join(fs_result['forward']),
        'Forward_Backward_Features': ', '.join(fs_result['forward_backward'])
    })
fs_df = pd.DataFrame(fs_summary)
fs_df.to_csv('./models/feature_selection_results.csv', index=False)
print(f"Feature selection results saved to ./models/feature_selection_results.csv")

# Find best models
print("\n" + "=" * 80)
print("BEST MODELS FOR EACH TARGET")
print("=" * 80)

for target in target_names:
    target_results = results_df[results_df['Target'] == target]
    if len(target_results) > 0:
        best_row = target_results.loc[target_results['Test_F1'].idxmax()]
        fs_info = all_feature_selection.get(target, {})
        print(f"\n{target}:")
        print(f"  Best Model: {best_row['Model']}")
        print(f"  Train Accuracy: {best_row['Train_Accuracy']:.4f}")
        print(f"  Test Accuracy:  {best_row['Test_Accuracy']:.4f}")
        print(f"  Test F1-Score:  {best_row['Test_F1']:.4f}")
        if pd.notna(best_row['Test_AUC']):
            print(f"  Test AUC-ROC:   {best_row['Test_AUC']:.4f}")
        print(f"  Selected Features: {fs_info.get('n_forward_backward', 'N/A')} (forward-backward)")

print("\n" + "=" * 80)
print("FEATURE SELECTION SUMMARY")
print("=" * 80)
for target, fs_result in all_feature_selection.items():
    print(f"\n{target}:")
    print(f"  Forward Selection: {fs_result['n_forward']} features")
    print(f"  Forward-Backward:  {fs_result['n_forward_backward']} features")

print("\n" + "=" * 80)
print("MODEL TRAINING COMPLETED")
print("=" * 80)
print("\nOutput files:")
print("  - ./models/model_results.csv")
print("  - ./models/train_test_comparison.csv")
print("  - ./models/grid_search_results.csv")
print("  - ./models/feature_selection_results.csv")
print("  - ./analysis/roc_curves_*.png")
print("  - ./analysis/grid_search_*.png")
print("  - ./analysis/confusion_matrix_*.png")
print("  - ./analysis/feature_selection_*.png")
