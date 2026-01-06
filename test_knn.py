import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, SplineTransformer
# --- 改动 1: 引入 KNNImputer ---
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
# 引入 Boosting 模块
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
import warnings
import os

warnings.filterwarnings('ignore')

# 确保输出目录存在
os.makedirs('./analysis', exist_ok=True)
os.makedirs('./models', exist_ok=True)

print("=" * 80)
print("CHILD APPENDICITIS PREDICTION - BOOSTING & ADDITIVE MODELS with KNN")
print("=" * 80)

# -------------------------------------------------------------------------
# 1. 路径配置 (保持您的本地路径)
# -------------------------------------------------------------------------
base_data_path = "./processed_data"  # 修改为包含所有特征数据的路径
save_path = "./models/boosting_knn_results.csv"  # 修改文件名以区分

print(f"\nLoading data from: {base_data_path}")

try:
    X_train = np.load(f"{base_data_path}/X_train_no_ultrasound.npy",
                      allow_pickle=True)
    X_test = np.load(f"{base_data_path}/X_test_no_ultrasound.npy",
                     allow_pickle=True)
    y_train_all = np.load(f"{base_data_path}/y_train_no_ultrasound.npy",
                          allow_pickle=True)
    y_test_all = np.load(f"{base_data_path}/y_test_no_ultrasound.npy",
                         allow_pickle=True)
    target_names = np.load(f"{base_data_path}/target_names.npy",
                           allow_pickle=True)

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Targets: {target_names}")

except FileNotFoundError as e:
    print(f"\n[Error] 找不到文件，请检查路径是否正确: {e}")
    exit()


# -------------------------------------------------------------------------
# 2. 定义模型工厂 (KNN + 加性模型 + Boosting)
# -------------------------------------------------------------------------
def get_boosting_models_with_knn():
    """
    返回 KNN 插补后的 Boosting 和加性模型。
    """

    # --- 改动 2: KNN 配置 ---
    # 使用 KNNImputer
    # n_neighbors=5: 找最近的5个邻居来投票/取平均
    knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')

    models = {
        # --- 基准加性模型 ---
        'Logistic Regression (GAM-Spline)':
        Pipeline([('imputer', knn_imputer), ('scaler', StandardScaler()),
                  ('spline',
                   SplineTransformer(n_knots=3, degree=2, include_bias=False)),
                  ('clf',
                   LogisticRegression(max_iter=1000, C=0.5, solver='saga'))]),

        # --- Boosting 模型 1: Gradient Boosting (GBDT) ---
        'Gradient Boosting (GBDT)':
        Pipeline([
            ('imputer', knn_imputer),  # 使用 KNN 填补
            ('scaler', StandardScaler()),
            ('clf',
             GradientBoostingClassifier(n_estimators=200,
                                        learning_rate=0.05,
                                        max_depth=3,
                                        random_state=42))
        ]),

        # --- Boosting 模型 2: Hist Gradient Boosting ---
        'Hist Gradient Boosting':
        Pipeline([('imputer', knn_imputer), ('scaler', StandardScaler()),
                  ('clf',
                   HistGradientBoostingClassifier(max_iter=200,
                                                  learning_rate=0.05,
                                                  max_depth=5,
                                                  random_state=42,
                                                  l2_regularization=0.1))]),

        # --- Boosting 模型 3: AdaBoost ---
        'AdaBoost':
        Pipeline([('imputer', knn_imputer), ('scaler', StandardScaler()),
                  ('clf',
                   AdaBoostClassifier(n_estimators=100,
                                      learning_rate=1.0,
                                      random_state=42,
                                      algorithm='SAMME.R'))])
    }
    return models


# -------------------------------------------------------------------------
# 3. 评估函数
# -------------------------------------------------------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(
        model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy

    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5

    return {
        'Error Rate': error_rate,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'AUC-ROC': auc
    }


# -------------------------------------------------------------------------
# 4. 主训练循环
# -------------------------------------------------------------------------
results_list = []

print("\n" + "=" * 80)
print("STARTING TRAINING (Boosting with KNN Imputation)")
print("=" * 80)

for i, target_name in enumerate(target_names):
    print(f"\nTarget: {target_name}")
    print("-" * 40)

    y_train = y_train_all[:, i]
    y_test = y_test_all[:, i]

    models = get_boosting_models_with_knn()

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        try:
            metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

            print(
                f"  > Error Rate: {metrics['Error Rate']:.4f} ({metrics['Error Rate']*100:.2f}%)"
            )
            print(f"  > F1-Score:   {metrics['F1-Score']:.4f}")
            print(f"  > AUC-ROC:    {metrics['AUC-ROC']:.4f}")

            res = metrics.copy()
            res['Model'] = model_name
            res['Target'] = target_name
            results_list.append(res)

        except Exception as e:
            print(f"  Error: {str(e)}")

# -------------------------------------------------------------------------
# 5. 结果汇总
# -------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FINAL SUMMARY (Sorted by Lowest Error Rate)")
print("=" * 80)

df_results = pd.DataFrame(results_list)

if not df_results.empty:
    # 按照 Error Rate 升序排列
    df_results = df_results.sort_values(by=['Target', 'Error Rate'],
                                        ascending=[True, True])

    cols = ['Target', 'Model', 'Error Rate', 'Accuracy', 'F1-Score', 'AUC-ROC']
    print(df_results[cols].to_string(index=False))

    df_results.to_csv(save_path, index=False)
    print(f"\nResults saved to {save_path}")
else:
    print("No results.")
