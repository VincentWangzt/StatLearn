import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, SplineTransformer
# 引入 MICE 模块
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression, BayesianRidge
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
print("CHILD APPENDICITIS PREDICTION - BOOSTING & ADDITIVE MODELS with MICE")
print("=" * 80)

# 1. 数据导入
print("\nLoading processed data...")

X_train = np.load('./processed_data/X_train_no_ultrasound.npy',
                  allow_pickle=True)
X_test = np.load('./processed_data/X_test_no_ultrasound.npy',
                 allow_pickle=True)
y_train_all = np.load('./processed_data/y_train_no_ultrasound.npy',
                      allow_pickle=True)
y_test_all = np.load('./processed_data/y_test_no_ultrasound.npy',
                     allow_pickle=True)
target_names = np.load('./processed_data/target_names.npy', allow_pickle=True)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")


# 2. 定义模型工厂 (MICE + 加性模型 + Boosting)
def get_boosting_models_with_mice():
    """
    返回 MICE 插补后的 Boosting 和加性模型。
    """

    # --- MICE 配置 ---
    # 这里使用 BayesianRidge (贝叶斯岭回归) 作为插补核心
    # 优点：计算快，假设数据之间存在线性关系，非常适合给 Boosting 提供基础数据
    mice_imputer = IterativeImputer(estimator=BayesianRidge(),
                                    max_iter=20,
                                    random_state=42,
                                    initial_strategy='median')

    models = {
        # --- 基准加性模型 (作为对比) ---
        'Logistic Regression (GAM-Spline)':
        Pipeline([('imputer', mice_imputer), ('scaler', StandardScaler()),
                  ('spline',
                   SplineTransformer(n_knots=3, degree=2, include_bias=False)),
                  ('clf',
                   LogisticRegression(max_iter=1000, C=0.5, solver='saga'))]),

        # --- Boosting 模型 1: 标准梯度提升 (GBDT) ---
        # 经典的加性模型：一步步修正残差
        'Gradient Boosting (GBDT)':
        Pipeline([
            ('imputer', mice_imputer),  # GBDT 不支持缺失值，必须先 MICE
            ('scaler', StandardScaler()),
            (
                'clf',
                GradientBoostingClassifier(
                    n_estimators=200,  # 树的数量
                    learning_rate=0.05,  # 学习率 (越低越稳，但需要更多树)
                    max_depth=3,  # 树深 (控制复杂度)
                    random_state=42))
        ]),

        # --- Boosting 模型 2: 直方图梯度提升 (新一代神器) ---
        # 类似于 LightGBM/XGBoost，速度极快，精度通常更高
        'Hist Gradient Boosting':
        Pipeline([
            ('imputer', mice_imputer),  # 虽然它原生支持NaN，但用MICE可以提供额外信息
            ('scaler', StandardScaler()),
            ('clf',
             HistGradientBoostingClassifier(max_iter=200,
                                            learning_rate=0.05,
                                            max_depth=5,
                                            random_state=42,
                                            l2_regularization=0.1))
        ]),

        # --- Boosting 模型 3: AdaBoost (元老级算法) ---
        # 关注那些被分错的样本，通过加权投票实现加性
        'AdaBoost':
        Pipeline([
            ('imputer', mice_imputer),
            ('scaler', StandardScaler()),
            (
                'clf',
                AdaBoostClassifier(
                    n_estimators=100,
                    learning_rate=1.0,
                    random_state=42,
                    algorithm='SAMME.R'  # 概率型提升
                ))
        ])
    }
    return models


# 3. 评估函数 (含错误率)
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(
        model, 'predict_proba') else None

    accuracy = accuracy_score(y_test, y_pred)
    error_rate = 1 - accuracy  # 错误率

    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.5

    return {
        'Error Rate': error_rate,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'AUC-ROC': auc
    }


# 4. 主训练循环
results_list = []

print("\n" + "=" * 80)
print("STARTING TRAINING (Boosting Enabled)")
print("=" * 80)

for i, target_name in enumerate(target_names):
    print(f"\nTarget: {target_name}")
    print("-" * 40)

    y_train = y_train_all[:, i]
    y_test = y_test_all[:, i]

    models = get_boosting_models_with_mice()

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

# 5. 结果汇总
print("\n" + "=" * 80)
print("FINAL SUMMARY (Sorted by Lowest Error Rate)")
print("=" * 80)

df_results = pd.DataFrame(results_list)

if not df_results.empty:
    # 按照 Error Rate 升序排列 (越小越好)
    df_results = df_results.sort_values(by=['Target', 'Error Rate'],
                                        ascending=[True, True])

    cols = ['Target', 'Model', 'Error Rate', 'Accuracy', 'F1-Score', 'AUC-ROC']
    print(df_results[cols].to_string(index=False))

    df_results.to_csv('./models/boosting_mice_results.csv', index=False)
    print(f"\nResults saved to ./models/boosting_mice_results.csv")
else:
    print("No results.")
