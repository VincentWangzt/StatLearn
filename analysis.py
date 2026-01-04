"""
Preliminary Analysis Script for Child Appendicitis ML Project
Generates correlation plots and visualizations for both:
- All features dataset
- No ultrasound dataset (also excludes Length_of_Stay)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure analysis directory exists
os.makedirs('./analysis', exist_ok=True)

# Set plot style
plt.style.use('seaborn-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def run_analysis(X, y, feature_names, target_names, suffix='', title_prefix=''):
    """
    Run complete analysis on a dataset.
    
    Args:
        X: Feature matrix
        y: Target matrix
        feature_names: Array of feature names
        target_names: Array of target names
        suffix: Suffix for output files (e.g., '_all_features', '_no_ultrasound')
        title_prefix: Prefix for plot titles (e.g., 'All Features: ', 'No Ultrasound: ')
    """
    print(f"\n{'='*60}")
    print(f"RUNNING ANALYSIS: {title_prefix.strip() if title_prefix else 'Default'}")
    print(f"{'='*60}")
    
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    print(f"Feature names: {len(feature_names)}")
    print(f"Target names: {list(target_names)}")
    
    # Create DataFrame for analysis
    df = pd.DataFrame(X, columns=feature_names)
    for i, target in enumerate(target_names):
        df[target] = y[:, i]
    
    print("\nGenerating visualizations...")
    
    # 1. Correlation Heatmap (Features vs Targets)
    print("1. Creating correlation heatmap...")
    # Select important features for the heatmap (to avoid cluttering)
    important_features = [
        'Age', 'BMI', 'Sex', 'Height', 'Weight', 'Length_of_Stay',
        'Alvarado_Score', 'Paedriatic_Appendicitis_Score', 'Body_Temperature',
        'WBC_Count', 'Neutrophil_Percentage', 'CRP', 'Neutrophilia', 'Hemoglobin',
        'RDW'
    ]
    
    # Keep features that exist
    important_features = [f for f in important_features if f in df.columns]
    analysis_cols = important_features + list(target_names)
    df_subset = df[analysis_cols]
    
    plt.figure(figsize=(14, 10))
    corr_matrix = df_subset.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5)
    plt.title(f'{title_prefix}Correlation Heatmap: Key Features and Target Variables',
              fontsize=14)
    plt.tight_layout()
    plt.savefig(f'./analysis/correlation_heatmap{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Full correlation heatmap with targets
    print("2. Creating target correlation analysis...")
    df_numeric = df.select_dtypes(include=[np.number])
    target_corr = df_numeric.corr()[list(target_names)].drop(list(target_names),
                                                             errors='ignore')
    target_corr = target_corr.dropna(how='all')
    target_corr_sorted = target_corr.reindex(
        target_corr['Diagnosis'].abs().sort_values(ascending=False).index)
    
    plt.figure(figsize=(10, 16))
    sns.heatmap(target_corr_sorted,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.5)
    plt.title(f'{title_prefix}Feature Correlations with Target Variables', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'./analysis/feature_target_correlations{suffix}.png',
                dpi=150,
                bbox_inches='tight')
    plt.close()
    
    # 3. Target Variable Distribution
    print("3. Creating target distribution plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, target in enumerate(target_names):
        ax = axes[i]
        counts = df[target].value_counts().sort_index()
        colors = sns.color_palette("husl", len(counts))
        bars = ax.bar(counts.index.astype(str), counts.values, color=colors)
        ax.set_title(f'{target} Distribution', fontsize=12)
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5,
                    str(val),
                    ha='center',
                    va='bottom',
                    fontsize=10)
    
    plt.suptitle(f'{title_prefix}Target Distributions', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'./analysis/target_distributions{suffix}.png',
                dpi=150,
                bbox_inches='tight')
    plt.close()
    
    # 4. Box plots: Key numerical features vs Diagnosis
    print("4. Creating box plots for key features...")
    numerical_features = [
        'Age', 'BMI', 'Body_Temperature', 'WBC_Count', 'Neutrophil_Percentage',
        'CRP', 'Alvarado_Score', 'Paedriatic_Appendicitis_Score'
    ]
    numerical_features = [f for f in numerical_features if f in df.columns]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(numerical_features):
        ax = axes[i]
        df.boxplot(column=feature, by='Diagnosis', ax=ax)
        ax.set_title(f'{feature} by Diagnosis')
        ax.set_xlabel('Diagnosis (0: No App, 1: App)')
        ax.set_ylabel(feature)
    
    plt.suptitle(f'{title_prefix}Key Features Distribution by Diagnosis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'./analysis/features_by_diagnosis{suffix}.png',
                dpi=150,
                bbox_inches='tight')
    plt.close()
    
    # 5. Box plots: Key numerical features vs Severity
    print("5. Creating box plots for severity...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(numerical_features):
        ax = axes[i]
        df.boxplot(column=feature, by='Severity', ax=ax)
        ax.set_title(f'{feature} by Severity')
        ax.set_xlabel('Severity (0: Uncomplicated, 1: Complicated)')
        ax.set_ylabel(feature)
    
    plt.suptitle(f'{title_prefix}Key Features Distribution by Severity', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'./analysis/features_by_severity{suffix}.png',
                dpi=150,
                bbox_inches='tight')
    plt.close()
    
    # 6. Box plots: Key numerical features vs Management
    print("6. Creating box plots for management...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(numerical_features):
        ax = axes[i]
        df.boxplot(column=feature, by='Management', ax=ax)
        ax.set_title(f'{feature} by Management')
        ax.set_xlabel('Management')
        ax.set_ylabel(feature)
    
    plt.suptitle(f'{title_prefix}Key Features Distribution by Management', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'./analysis/features_by_management{suffix}.png',
                dpi=150,
                bbox_inches='tight')
    plt.close()
    
    # 7. Scatter plot matrix for key features
    print("7. Creating scatter plot matrix...")
    scatter_features = ['Age', 'WBC_Count', 'CRP', 'Alvarado_Score', 'Diagnosis']
    scatter_features = [f for f in scatter_features if f in df.columns]
    
    fig = plt.figure(figsize=(12, 12))
    g = sns.pairplot(df[scatter_features],
                     hue='Diagnosis',
                     diag_kind='hist',
                     plot_kws={'alpha': 0.6},
                     palette='husl')
    g.fig.suptitle(f'{title_prefix}Scatter Plot Matrix: Key Features', y=1.02)
    plt.savefig(f'./analysis/scatter_matrix{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 8. Top correlated features bar chart
    print("8. Creating top correlations bar chart...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, target in enumerate(target_names):
        ax = axes[i]
        corr_with_target = df.corr()[target].drop(list(target_names))
        top_corr = corr_with_target.abs().sort_values(ascending=False).head(15)
        top_corr_values = corr_with_target[top_corr.index]
        
        colors = ['green' if v > 0 else 'red' for v in top_corr_values]
        bars = ax.barh(range(len(top_corr)),
                       top_corr_values,
                       color=colors,
                       alpha=0.7)
        ax.set_yticks(range(len(top_corr)))
        ax.set_yticklabels(top_corr.index)
        ax.set_xlabel('Correlation Coefficient')
        ax.set_title(f'Top 15 Features Correlated with {target}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.invert_yaxis()
    
    plt.suptitle(f'{title_prefix}Top Correlations', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'./analysis/top_correlations{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 9. Class balance visualization
    print("9. Creating class balance pie charts...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    target_labels = {
        'Diagnosis': {
            0: 'No Appendicitis',
            1: 'Appendicitis'
        },
        'Management': {
            0: 'Conservative',
            1: 'Primary Surgical',
            2: 'Secondary Surgical'
        },
        'Severity': {
            0: 'Uncomplicated/None',
            1: 'Complicated'
        }
    }
    
    for i, target in enumerate(target_names):
        ax = axes[i]
        counts = df[target].value_counts().sort_index()
        labels = [target_labels[target].get(idx, str(idx)) for idx in counts.index]
        colors = sns.color_palette("pastel", len(counts))
        
        wedges, texts, autotexts = ax.pie(counts,
                                          labels=labels,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90)
        ax.set_title(f'{target} Class Distribution')
    
    plt.suptitle(f'{title_prefix}Class Balance', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f'./analysis/class_balance{suffix}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 10. Heatmap of binary features vs targets
    print("10. Creating binary features analysis...")
    binary_features = [
        'Sex', 'Migratory_Pain', 'Lower_Right_Abd_Pain',
        'Contralateral_Rebound_Tenderness', 'Coughing_Pain', 'Nausea',
        'Loss_of_Appetite', 'Neutrophilia', 'Dysuria', 'Psoas_Sign',
        'Ipsilateral_Rebound_Tenderness'
    ]
    binary_features = [f for f in binary_features if f in df.columns]
    
    if len(binary_features) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, target in enumerate(target_names):
            ax = axes[i]
            contingency_data = {}
            for feat in binary_features:
                unique_vals = df[feat].dropna().unique()
                for val in unique_vals:
                    subset = df[df[feat] == val]
                    valid_subset = subset[subset[target].notna()]
                    if len(valid_subset) > 0:
                        pct = (valid_subset[target] == 1).mean() * 100
                        contingency_data[f"{feat}={int(val)}"] = pct
            
            features_list = list(contingency_data.keys())
            values = list(contingency_data.values())
            
            colors = plt.cm.RdYlGn(np.array(values) / 100)
            ax.barh(features_list, values, color=colors)
            ax.set_xlabel(f'% with {target}=1')
            ax.set_title(f'Binary Features vs {target}')
            ax.set_xlim(0, 100)
        
        plt.suptitle(f'{title_prefix}Binary Features Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f'./analysis/binary_features_analysis{suffix}.png',
                    dpi=150,
                    bbox_inches='tight')
        plt.close()
    
    print(f"\nAnalysis complete for {title_prefix.strip() if title_prefix else 'dataset'}!")
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("CHILD APPENDICITIS PROJECT - DATA ANALYSIS")
    print("=" * 80)
    
    # Load target names (same for both datasets)
    target_names = np.load('./processed_data/target_names.npy', allow_pickle=True)
    
    # ==========================================================================
    # ANALYSIS 1: ALL FEATURES DATASET
    # ==========================================================================
    print("\n" + "=" * 80)
    print("LOADING ALL FEATURES DATASET")
    print("=" * 80)
    
    X_all = np.load('./processed_data/X_all_features.npy', allow_pickle=True)
    y_all = np.load('./processed_data/y_all_features.npy', allow_pickle=True)
    feature_names_all = np.load('./processed_data/feature_names_all.npy', allow_pickle=True)
    
    df_all = run_analysis(
        X_all, y_all, feature_names_all, target_names,
        suffix='_all_features',
        title_prefix='All Features: '
    )
    
    # ==========================================================================
    # ANALYSIS 2: NO ULTRASOUND DATASET (also excludes Length_of_Stay)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("LOADING NO ULTRASOUND DATASET")
    print("(Also excludes Length_of_Stay as leaky feature)")
    print("=" * 80)
    
    X_no_us = np.load('./processed_data/X_no_ultrasound.npy', allow_pickle=True)
    y_no_us = np.load('./processed_data/y_no_ultrasound.npy', allow_pickle=True)
    feature_names_no_us = np.load('./processed_data/feature_names_no_ultrasound.npy', allow_pickle=True)
    
    df_no_us = run_analysis(
        X_no_us, y_no_us, feature_names_no_us, target_names,
        suffix='_no_ultrasound',
        title_prefix='No Ultrasound: '
    )
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    
    print("\nDataset Summary:")
    print(f"  All Features: {X_all.shape[1]} features, {X_all.shape[0]} samples")
    print(f"  No Ultrasound: {X_no_us.shape[1]} features, {X_no_us.shape[0]} samples")
    
    print("\nGenerated plots saved in ./analysis/ directory:")
    print("\n  All Features (*_all_features.png):")
    print("    - correlation_heatmap_all_features.png")
    print("    - feature_target_correlations_all_features.png")
    print("    - target_distributions_all_features.png")
    print("    - features_by_diagnosis_all_features.png")
    print("    - features_by_severity_all_features.png")
    print("    - features_by_management_all_features.png")
    print("    - scatter_matrix_all_features.png")
    print("    - top_correlations_all_features.png")
    print("    - class_balance_all_features.png")
    print("    - binary_features_analysis_all_features.png")
    
    print("\n  No Ultrasound (*_no_ultrasound.png):")
    print("    - correlation_heatmap_no_ultrasound.png")
    print("    - feature_target_correlations_no_ultrasound.png")
    print("    - target_distributions_no_ultrasound.png")
    print("    - features_by_diagnosis_no_ultrasound.png")
    print("    - features_by_severity_no_ultrasound.png")
    print("    - features_by_management_no_ultrasound.png")
    print("    - scatter_matrix_no_ultrasound.png")
    print("    - top_correlations_no_ultrasound.png")
    print("    - class_balance_no_ultrasound.png")
    print("    - binary_features_analysis_no_ultrasound.png")
