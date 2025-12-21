"""
Preliminary Analysis Script for Child Appendicitis ML Project
Generates correlation plots and visualizations
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

# Load processed data
print("Loading processed data...")
X_all = np.load('./processed_data/X_all_features.npy', allow_pickle=True)
y_all = np.load('./processed_data/y_all_features.npy', allow_pickle=True)
feature_names = np.load('./processed_data/feature_names_all.npy',
                        allow_pickle=True)
target_names = np.load('./processed_data/target_names.npy', allow_pickle=True)

print(f"Features shape: {X_all.shape}")
print(f"Targets shape: {y_all.shape}")
print(f"Feature names: {len(feature_names)}")
print(f"Target names: {target_names}")

# Create DataFrame for analysis
df = pd.DataFrame(X_all, columns=feature_names)
for i, target in enumerate(target_names):
    df[target] = y_all[:, i]

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
plt.title('Correlation Heatmap: Key Features and Target Variables',
          fontsize=14)
plt.tight_layout()
plt.savefig('./analysis/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. Full correlation heatmap with targets
print("2. Creating target correlation analysis...")
# Compute correlation on numeric columns, handling NaN
df_numeric = df.select_dtypes(include=[np.number])
target_corr = df_numeric.corr()[list(target_names)].drop(list(target_names),
                                                         errors='ignore')
target_corr = target_corr.dropna(how='all')  # Remove rows with all NaN
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
plt.title('Feature Correlations with Target Variables', fontsize=14)
plt.tight_layout()
plt.savefig('./analysis/feature_target_correlations.png',
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

    # Add value labels on bars
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(val),
                ha='center',
                va='bottom',
                fontsize=10)

plt.tight_layout()
plt.savefig('./analysis/target_distributions.png',
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

plt.suptitle('Key Features Distribution by Diagnosis', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('./analysis/features_by_diagnosis.png',
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

plt.suptitle('Key Features Distribution by Severity', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('./analysis/features_by_severity.png',
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

plt.suptitle('Key Features Distribution by Management', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('./analysis/features_by_management.png',
            dpi=150,
            bbox_inches='tight')
plt.close()

# 7. Scatter plot matrix for key features
print("7. Creating scatter plot matrix...")
scatter_features = ['Age', 'WBC_Count', 'CRP', 'Alvarado_Score', 'Diagnosis']
scatter_features = [f for f in scatter_features if f in df.columns]

fig = plt.figure(figsize=(12, 12))
sns.pairplot(df[scatter_features],
             hue='Diagnosis',
             diag_kind='hist',
             plot_kws={'alpha': 0.6},
             palette='husl')
plt.suptitle('Scatter Plot Matrix: Key Features', y=1.02)
plt.savefig('./analysis/scatter_matrix.png', dpi=150, bbox_inches='tight')
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

plt.tight_layout()
plt.savefig('./analysis/top_correlations.png', dpi=150, bbox_inches='tight')
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

plt.tight_layout()
plt.savefig('./analysis/class_balance.png', dpi=150, bbox_inches='tight')
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
        # Create contingency table
        contingency_data = {}
        for feat in binary_features:
            # Calculate percentage of each class for each feature value
            # Skip NaN values
            unique_vals = df[feat].dropna().unique()
            for val in unique_vals:
                subset = df[df[feat] == val]
                # Also skip NaN in target
                valid_subset = subset[subset[target].notna()]
                if len(valid_subset) > 0:
                    pct = (valid_subset[target] == 1).mean() * 100
                    contingency_data[f"{feat}={int(val)}"] = pct

        # Plot as horizontal bar chart
        features_list = list(contingency_data.keys())
        values = list(contingency_data.values())

        colors = plt.cm.RdYlGn(np.array(values) / 100)
        ax.barh(features_list, values, color=colors)
        ax.set_xlabel(f'% with {target}=1')
        ax.set_title(f'Binary Features vs {target}')
        ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig('./analysis/binary_features_analysis.png',
                dpi=150,
                bbox_inches='tight')
    plt.close()

print("\nAll visualizations saved in ./analysis/ directory!")
print("Generated plots:")
print("  1. correlation_heatmap.png - Overall correlation matrix")
print("  2. feature_target_correlations.png - Features vs targets")
print("  3. target_distributions.png - Target variable distributions")
print("  4. features_by_diagnosis.png - Box plots by diagnosis")
print("  5. features_by_severity.png - Box plots by severity")
print("  6. features_by_management.png - Box plots by management")
print("  7. scatter_matrix.png - Scatter plot matrix")
print("  8. top_correlations.png - Top correlated features")
print("  9. class_balance.png - Class balance pie charts")
print("  10. binary_features_analysis.png - Binary features analysis")
