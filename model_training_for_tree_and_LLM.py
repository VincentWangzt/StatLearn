import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
import warnings
import os
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import re
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from StatLearn.llm import LLM
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


llm_ds = LLM(
    model_name="deepseek-chat",
    llm_url="https://api.deepseek.com/chat/completions",
    api_key="sk-", # Need to change to your api
    format="openai",
)
llm_gpt = LLM(
    model_name="gpt-5-2025-08-07",
    llm_url="https://api.aimlapi.com/v1/chat/completions",
    api_key="", # Need to change to your api, I use aimlapi here
    format="openai",
    proxy_url="http://127.0.0.1:7897", # VPN port
)
class LLMClassifier(BaseEstimator, ClassifierMixin):
    """
    一个 sklearn 兼容的“假”分类器：
    - fit 时：只记录标签集合 + 多数类 + 若干代表性训练样本（few-shot examples）
    - predict 时：把 few-shot + 当前病人特征一起喂给 LLM，让它输出一个整数类标。
    - 支持在训练集上跑一遍，利用错分样本迭代更新 few-shot。
    - 新增：内部训练若干 baseline 模型，并在 prompt 里加入它们对当前病人的预测。
    """

    def __init__(
        self,
        llm,
        feature_names,
        target_name,
        id2label=None,
        n_shots_per_class=3,
        max_total_shots=8,
        role_description=None,
        # few-shot 迭代 refinement
        refine_rounds=1,
        refine_max_errors_per_round=32,
        refine_random_state=42,
        refine_train_subset_size=128,
        # 🔴 新增：baseline 模型相关参数
        base_estimators=None,
        use_base_proba=True,
        feature_explanations=None,
    ):
        # 这些都是“超参数”，不能在 __init__ 里改形态
        self.llm = llm
        self.feature_names = feature_names
        self.target_name = target_name
        self.id2label = id2label
        self.n_shots_per_class = n_shots_per_class
        self.max_total_shots = max_total_shots
        self.role_description = role_description
        self.refine_train_subset_size = refine_train_subset_size

        # refinement 控制
        self.refine_rounds = refine_rounds
        self.refine_max_errors_per_round = refine_max_errors_per_round
        self.refine_random_state = refine_random_state

        # 🔴 新增：baseline 模型配置
        self.base_estimators = base_estimators
        self.use_base_proba = use_base_proba

        # 训练后才有的属性
        self.classes_ = None
        self.majority_class_ = None
        self.few_shot_examples = []

        # 派生内部属性
        self._feature_names_ = None
        self._id2label_ = None

        # 🔴 新增：内部 baseline 模型 + imputer
        self.base_models_ = None
        self.imputer_ = None
        
        self.feature_explanations = feature_explanations or {}

    # ------------------------------------------------------------------
    # 训练阶段
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_, counts = np.unique(y, return_counts=True)
        self.majority_class_ = self.classes_[np.argmax(counts)]

        # 安全部分
        self._feature_names_ = list(self.feature_names)
        self._id2label_ = dict(self.id2label) if self.id2label is not None else {}

        # 🔴 1) 先为 baseline 模型准备 imputer + 训练
        from sklearn.impute import SimpleImputer

        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X)

        self.base_models_ = None
        if self.base_estimators is not None and len(self.base_estimators) > 0:
            self.base_models_ = {}
            for name, est in self.base_estimators.items():
                m = clone(est)
                m.fit(X_imp, y)
                self.base_models_[name] = m

        # 2) 按“类中心”构造初始 few-shot（这里仍然使用原始 X，内部会自己做均值填补）
        self._build_few_shot_examples(X, y)

        # 3) “错题驱动”刷新 few-shot
        self._refine_few_shots_with_errors(X, y)

        return self

    # ------------------------------------------------------------------
    # 构造初始 few-shot: 类中心附近
    # ------------------------------------------------------------------
    def _build_few_shot_examples(self, X, y):
        self.few_shot_examples = []

        if X.size == 0:
            return

        X_imp = X.copy()
        col_means = np.nanmean(X_imp, axis=0)
        inds = np.where(np.isnan(X_imp))
        X_imp[inds] = np.take(col_means, inds[1])

        for cls in self.classes_:
            idx = np.where(y == cls)[0]
            if len(idx) == 0:
                continue

            X_c = X_imp[idx]
            center = X_c.mean(axis=0)
            dists = np.linalg.norm(X_c - center, axis=1)

            k = min(self.n_shots_per_class, len(idx))
            chosen_local = np.argsort(dists)[:k]
            chosen_indices = idx[chosen_local]

            for j in chosen_indices:
                self.few_shot_examples.append({
                    "y": int(y[j]),
                    "x": X[j].copy()
                })

        if len(self.few_shot_examples) > self.max_total_shots:
            self.few_shot_examples = self.few_shot_examples[:self.max_total_shots]

    # ------------------------------------------------------------------
    # 错题驱动 few-shot 更新（保持原样）
    # ------------------------------------------------------------------
    def _update_few_shot_with_error(self, y_true, x_true):
        y_true = int(y_true)
        x_true = np.asarray(x_true, dtype=float).copy()

        same_class_indices = [i for i, ex in enumerate(self.few_shot_examples)
                              if ex["y"] == y_true]

        if len(same_class_indices) < self.n_shots_per_class:
            self.few_shot_examples.append({
                "y": y_true,
                "x": x_true
            })
        else:
            idx_replace = same_class_indices[0]
            self.few_shot_examples[idx_replace] = {
                "y": y_true,
                "x": x_true
            }

        if len(self.few_shot_examples) > self.max_total_shots:
            self.few_shot_examples = self.few_shot_examples[-self.max_total_shots:]

    def _refine_few_shots_with_errors(self, X, y):
        if self.refine_rounds <= 0:
            return

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        rng = np.random.RandomState(self.refine_random_state)
        n_samples = X.shape[0]

        for r in range(self.refine_rounds):
            print(f"[LLMClassifier] refine round {r+1}/{self.refine_rounds} ...")

            if self.refine_train_subset_size is not None \
               and self.refine_train_subset_size < n_samples:
                subset_size = self.refine_train_subset_size
                subset_idx = rng.choice(n_samples, size=subset_size, replace=False)
                X_sub = X[subset_idx]
                y_sub = y[subset_idx]
            else:
                subset_idx = np.arange(n_samples)
                X_sub = X
                y_sub = y

            y_pred_sub = self.predict(X_sub)
            mis_local = np.where(y_pred_sub != y_sub)[0]
            if mis_local.size == 0:
                print("[LLMClassifier] no misclassifications in subset, stop refinement.")
                break

            mis_idx = subset_idx[mis_local]
            rng.shuffle(mis_idx)
            mis_idx = mis_idx[: self.refine_max_errors_per_round]

            for i in mis_idx:
                self._update_few_shot_with_error(y_true=y[i], x_true=X[i])

    # ------------------------------------------------------------------
    # 文本化特征
    # ------------------------------------------------------------------
    def _case_to_text(self, x):
        parts = []
        for name, val in zip(self._feature_names_, x):
            # 查字典：Variable Name in Data Files -> Explanation
            desc = self.feature_explanations.get(name)
            if desc:
                name_str = f"{name} ({desc})"
            else:
                name_str = name

            if isinstance(val, (float, np.floating)) and np.isnan(val):
                v_str = "missing"
            else:
                v_str = str(val)

            parts.append(f"{name_str} = {v_str}")
        return "; ".join(parts)

    # 🔴 新增：baseline 模型预测汇总
    def _summarize_base_predictions(self, x):
        """
        针对单个样本 x，用内部 baseline 模型做预测，
        生成一段自然语言 summary，插入到 prompt 里。
        """
        if self.base_models_ is None or self.imputer_ is None:
            return ""

        x_arr = np.asarray(x, dtype=float).reshape(1, -1)
        x_imp = self.imputer_.transform(x_arr)

        lines = []
        for name, model in self.base_models_.items():
            if self.use_base_proba and hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_imp)[0]
                best_idx = int(np.argmax(proba))
                label_id = self.classes_[best_idx]
                label_name = self._id2label_.get(int(label_id), f"class_{int(label_id)}")
                prob_val = float(proba[best_idx])
                lines.append(
                    f"{name}: predicted {int(label_id)} ({label_name}), "
                    f"probability = {prob_val:.3f}"
                )
            else:
                pred = model.predict(x_imp)[0]
                label_name = self._id2label_.get(int(pred), f"class_{int(pred)}")
                lines.append(
                    f"{name}: predicted {int(pred)} ({label_name})"
                )

        if not lines:
            return ""

        text = (
            "Here are predictions from several baseline machine learning models "
            "trained on the same training data:\n"
            + "\n".join(f"- {ln}" for ln in lines)
            + "\n"
        )
        return text

    # ------------------------------------------------------------------
    # 构造 prompt（加入 baseline 预测）
    # ------------------------------------------------------------------
    def _build_prompt(self, x):
        case_text = self._case_to_text(x)

        class_ids = list(self.classes_)
        class_ids_str = ", ".join(str(int(c)) for c in class_ids)

        mapping_lines = []
        for cid in class_ids:
            cid_int = int(cid)
            label_name = self._id2label_.get(cid_int, f"class_{cid_int}")
            mapping_lines.append(f"{cid_int} = {label_name}")
        mapping_text = "\n".join(mapping_lines)

        # few-shot 例子
        examples_blocks = []
        for i, ex in enumerate(self.few_shot_examples):
            y_ex = ex["y"]
            x_ex = ex["x"]
            label_name = self._id2label_.get(y_ex, f"class_{y_ex}")
            ex_text = self._case_to_text(x_ex)
            block = (
                f"Example {i+1}:\n"
                f"Features: {ex_text}\n"
                f"Correct label for {self.target_name}: {y_ex} ({label_name})\n"
            )
            examples_blocks.append(block)
        examples_text = ""
        if examples_blocks:
            examples_text = (
                "Here are some example patients and their correct labels:\n\n"
                + "\n".join(examples_blocks)
                + "\n"
            )

        # baseline 模型预测文本
        base_pred_text = self._summarize_base_predictions(x)

        if self.role_description is not None:
            role_text = self.role_description.strip()
        else:
            role_text = (
                "You are a pediatric emergency surgeon. "
                "You will see clinical features of one child, and you must predict "
                f"the label for the target: \"{self.target_name}\".\n"
            )

        prompt = f"""
{role_text}

The target is encoded as integer labels. The meaning of each label is:

{mapping_text}

{base_pred_text}
{examples_text}
Now here is a NEW patient (different from all the examples above).

The features and their values of THIS child are:

{case_text}

Possible labels you are allowed to output are: [{class_ids_str}].

Please output ONLY a single integer from [{class_ids_str}] as your prediction,
with no extra words, no explanation.
"""
        return prompt

    # ------------------------------------------------------------------
    # LLM 输出解析 + 并行预测（保持原样）
    # ------------------------------------------------------------------
    def _parse_label(self, raw_text):
        if raw_text is None:
            return self.majority_class_

        text = str(raw_text).strip()
        m = re.search(r"-?\d+", text)
        if not m:
            return self.majority_class_

        try:
            pred = int(m.group(0))
        except ValueError:
            return self.majority_class_

        if pred in self.classes_:
            return pred
        else:
            return self.majority_class_

    def _predict_one(self, x):
        prompt = self._build_prompt(x)
        raw = self.llm.query(prompt)
        label = self._parse_label(raw)
        return label

    def predict(self, X):
        X = np.asarray(X)
        n_total = X.shape[0]
        preds = [None] * n_total

        max_workers = 16

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for i in range(n_total):
                x = X[i, :]
                f = ex.submit(self._predict_one, x)
                futures[f] = i

            for f in tqdm(as_completed(futures),
                          total=n_total,
                          desc="LLM test (parallel)"):
                i = futures[f]
                try:
                    preds[i] = f.result()
                except Exception:
                    preds[i] = self.majority_class_

        return np.asarray(preds)

class _SimpleMLP(nn.Module):
    """
    一个很小的 MLP，用于把原始特征映射到一个 hidden 表示，再接 RandomForest。
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, return_features=False):
        h = self.feature(x)
        if return_features:
            return h
        logits = self.classifier(h)
        return logits


class NeuralDecisionForest(BaseEstimator, ClassifierMixin):
    """
    一个非常简化版的 Neural Decision Forest：
    1. 用 MLP 先做一个 supervised 训练（交叉熵）。
    2. 用 MLP 的倒数第二层 hidden 表示，再训练一个 RandomForestClassifier。
    3. 预测时：X -> MLP(hidden) -> RandomForest.predict。
    """

    def __init__(
        self,
        hidden_dim=64,
        n_estimators=200,
        max_depth=None,
        batch_size=32,
        epochs=20,
        lr=1e-3,
        device=None,
        random_state=42,
    ):
        self.hidden_dim = hidden_dim
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.random_state = random_state

        # fit 之后才有这些属性
        self.input_dim_ = None
        self.n_classes_ = None
        self.mlp_ = None
        self.tree_ = None
        self.device_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        self.input_dim_ = X.shape[1]

        # 设备
        if self.device is not None:
            device = self.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device_ = torch.device(device)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # 构建 MLP
        self.mlp_ = _SimpleMLP(
            in_dim=self.input_dim_,
            hidden_dim=self.hidden_dim,
            out_dim=self.n_classes_,
        ).to(self.device_)

        # 准备数据
        X_tensor = torch.from_numpy(X).to(self.device_)
        # 假设 y 已经是 0..C-1 的整数编码
        y_tensor = torch.from_numpy(y.astype(np.int64)).to(self.device_)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # 训练 MLP
        optimizer = torch.optim.Adam(self.mlp_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        self.mlp_.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.mlp_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        # 抽取 hidden 表示，并训练 RandomForest
        self.mlp_.eval()
        with torch.no_grad():
            all_hidden = self.mlp_(X_tensor, return_features=True)
        hidden_np = all_hidden.cpu().numpy()

        self.tree_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.tree_.fit(hidden_np, y)

        return self

    def _transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.from_numpy(X).to(self.device_)
        self.mlp_.eval()
        with torch.no_grad():
            hidden = self.mlp_(X_tensor, return_features=True)
        hidden_np = hidden.cpu().numpy()
        return hidden_np

    def predict(self, X):
        hidden_np = self._transform(X)
        return self.tree_.predict(hidden_np)

    def predict_proba(self, X):
        hidden_np = self._transform(X)
        if hasattr(self.tree_, "predict_proba"):
            return self.tree_.predict_proba(hidden_np)
        # 兜底：如果没有 predict_proba（理论上有），就做 one-hot
        preds = self.tree_.predict(hidden_np)
        proba = np.zeros((len(preds), self.n_classes_), dtype=float)
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0][0]
            proba[i, idx] = 1.0
        return proba
    
class TreeEnsembleLayerClassifier(BaseEstimator, ClassifierMixin):
    """
    Tree Ensemble Layer (TEL) 近似实现：
    - 第一步：训练两个树模型（RandomForest + GradientBoosting）
    - 第二步：用它们的 predict_proba 作为“高层特征”
    - 第三步：把高层特征喂给 LogisticRegression 做最终分类
    """

    def __init__(
        self,
        rf_n_estimators=200,
        rf_max_depth=None,
        gb_n_estimators=200,
        gb_learning_rate=0.05,
        gb_max_depth=3,
        C=1.0,
        random_state=42,
    ):
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.gb_n_estimators = gb_n_estimators
        self.gb_learning_rate = gb_learning_rate
        self.gb_max_depth = gb_max_depth
        self.C = C
        self.random_state = random_state

        self.rf_ = None
        self.gb_ = None
        self.lr_ = None
        self.classes_ = None
        self.n_classes_ = None

    def _build_high_level_features(self, X):
        """
        用 base trees 的 predict_proba 生成“高层特征”：
        [rf_proba, gb_proba] 按列拼接
        """
        rf_proba = self.rf_.predict_proba(X)    # shape: (n_samples, n_classes)
        gb_proba = self.gb_.predict_proba(X)    # 同上
        return np.hstack([rf_proba, gb_proba])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # 1) base trees
        self.rf_ = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.gb_ = GradientBoostingClassifier(
            n_estimators=self.gb_n_estimators,
            learning_rate=self.gb_learning_rate,
            max_depth=self.gb_max_depth,
            random_state=self.random_state,
        )

        self.rf_.fit(X, y)
        self.gb_.fit(X, y)

        # 2) 高层特征
        H = self._build_high_level_features(X)

        # 3) 最后一层线性分类器
        self.lr_ = LogisticRegression(
            C=self.C,
            max_iter=1000,
            multi_class="auto",
            solver="lbfgs",
            random_state=self.random_state,
        )
        self.lr_.fit(H, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        H = self._build_high_level_features(X)
        return self.lr_.predict(H)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        H = self._build_high_level_features(X)
        if hasattr(self.lr_, "predict_proba"):
            return self.lr_.predict_proba(H)
        # 兜底：如果没有 predict_proba，就 one-hot
        preds = self.lr_.predict(H)
        proba = np.zeros((len(preds), self.n_classes_), dtype=float)
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0][0]
            proba[i, idx] = 1.0
        return proba


class TreeNetClassifier(BaseEstimator, ClassifierMixin):
    """
    一个简单版本的 TreeNet：
    - Level 1: RandomForest
    - Level 2: RandomForest, 输入是 [原始特征, Level1 的 predict_proba]
    """

    def __init__(
        self,
        level1_n_estimators=200,
        level1_max_depth=None,
        level2_n_estimators=200,
        level2_max_depth=None,
        random_state=42,
    ):
        self.level1_n_estimators = level1_n_estimators
        self.level1_max_depth = level1_max_depth
        self.level2_n_estimators = level2_n_estimators
        self.level2_max_depth = level2_max_depth
        self.random_state = random_state

        self.rf1_ = None
        self.rf2_ = None
        self.classes_ = None
        self.n_classes_ = None

    def _concat_with_level1_proba(self, X):
        """
        生成 TreeNet 第二层的输入：
        Z = [X, rf1.predict_proba(X)]
        """
        proba1 = self.rf1_.predict_proba(X)
        return np.hstack([X, proba1])

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Level 1
        self.rf1_ = RandomForestClassifier(
            n_estimators=self.level1_n_estimators,
            max_depth=self.level1_max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rf1_.fit(X, y)

        # 拼接 Level1 输出
        Z = self._concat_with_level1_proba(X)

        # Level 2
        self.rf2_ = RandomForestClassifier(
            n_estimators=self.level2_n_estimators,
            max_depth=self.level2_max_depth,
            random_state=self.random_state + 1,
            n_jobs=-1,
        )
        self.rf2_.fit(Z, y)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Z = self._concat_with_level1_proba(X)
        return self.rf2_.predict(Z)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        Z = self._concat_with_level1_proba(X)
        if hasattr(self.rf2_, "predict_proba"):
            return self.rf2_.predict_proba(Z)
        # 兜底
        preds = self.rf2_.predict(Z)
        proba = np.zeros((len(preds), self.n_classes_), dtype=float)
        for i, p in enumerate(preds):
            idx = np.where(self.classes_ == p)[0][0]
            proba[i, idx] = 1.0
        return proba
    
class BoundaryLLMHybrid(BaseEstimator, ClassifierMixin):
    """
    Boundary LLM Hybrid：
    - 多个树模型组成一个 committee，在训练集上定义“边界样本”(uncertain samples)
    - 对于边界样本，交给 LLMClassifier 重新评估
    - 对于非边界样本，直接用 base_estimator 的预测
    """

    def __init__(
        self,
        base_estimator=None,
        boundary_quantile=0.2,
        llm_estimator=None,
        random_state=42,
    ):
        """
        base_estimator: 负责大部分样本的基础模型（默认用一个 RF）
        boundary_quantile: 取多少比例的“最不确定”样本作为边界样本（比如 0.2 = top 20%）
        llm_estimator: 一个已经配置好 feature_names / target_name / id2label 的 LLMClassifier 实例
        """
        self.base_estimator = base_estimator
        self.boundary_quantile = boundary_quantile
        self.llm_estimator = llm_estimator
        self.random_state = random_state

        # 训练后才有的属性
        self.classes_ = None
        self.n_classes_ = None
        self.base_ = None
        self.committee_ = None
        self.llm_ = None
        self.threshold_ = None

        # 先定义好 committee“模板”三个树模型（偏强 & 风格不同）
        self._committee_templates = [
            ("rf", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1,
            )),
            ("gb", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state + 1,
            )),
            ("et", ExtraTreesClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=self.random_state + 2,
                n_jobs=-1,
            )),
        ]

    # -------------------------
    # 内部：计算不确定度
    # -------------------------
    def _compute_uncertainty(self, X, alpha=1.0):
        """
        alpha: 你现在只用概率型不确定度；后面想加分歧可以再调
        """
        X = np.asarray(X, dtype=float)
        proba_list = []

        for name, est in self.committee_:
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X)
            else:
                pred = est.predict(X)
                p = np.zeros((len(pred), self.n_classes_), dtype=float)
                for i, y in enumerate(pred):
                    idx = np.where(self.classes_ == y)[0][0]
                    p[i, idx] = 1.0
            proba_list.append(p)

        proba_arr = np.stack(proba_list, axis=0)   # (n_models, n_samples, n_classes)
        max_prob = proba_arr.max(axis=2)           # (n_models, n_samples)
        mean_max_prob = max_prob.mean(axis=0)      # (n_samples,)

        # 不确定度 = 1 - 平均信心
        uncert = 1.0 - mean_max_prob
        return uncert

    # -------------------------
    # fit
    # -------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # 1) base 模型：默认是一个 RF
        if self.base_estimator is None:
            base = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            base = clone(self.base_estimator)
        base.fit(X, y)
        self.base_ = base

        # 2) 训练 committee 的多个树模型
        self.committee_ = []
        for name, tmpl in self._committee_templates:
            est = clone(tmpl)
            est.fit(X, y)
            self.committee_.append((name, est))

        # 3) 在训练集上计算不确定度 & 选出边界样本
        uncert_train = self._compute_uncertainty(X)
        self.threshold_ = np.quantile(
            uncert_train, 1.0 - self.boundary_quantile
        )  # top boundary_quantile as boundary
        boundary_mask = uncert_train > self.threshold_
        X_boundary = X[boundary_mask]
        y_boundary = y[boundary_mask]

        # 4) 训练 LLMClassifier（只用边界样本做 few-shot 原型）
        if self.llm_estimator is not None and X_boundary.shape[0] > 0:
            self.llm_ = clone(self.llm_estimator)
            self.llm_.fit(X_boundary, y_boundary)
        else:
            self.llm_ = None

        return self

    # -------------------------
    # predict
    # -------------------------
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        # 先用 base 模型预测
        base_pred = self.base_.predict(X)

        # 没有 LLM，就直接返回
        if self.llm_ is None or self.committee_ is None or self.threshold_ is None:
            return base_pred

        # 用 committee 评估当前 X 的不确定度
        uncert = self._compute_uncertainty(X)
        boundary_mask = uncert > self.threshold_

        # 对于边界样本，用 LLM 的预测替换
        if boundary_mask.any():
            X_boundary = X[boundary_mask]
            llm_pred = self.llm_.predict(X_boundary)
            base_pred[boundary_mask] = llm_pred

        return base_pred

    # -------------------------
    # predict_proba（粗略版，用 one-hot 近似 LLM）
    # -------------------------
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if hasattr(self.base_, "predict_proba"):
            proba = self.base_.predict_proba(X)
        else:
            # 如果 base 没有 proba，退化成 one-hot
            base_pred = self.base_.predict(X)
            proba = np.zeros((len(base_pred), self.n_classes_), dtype=float)
            for i, y in enumerate(base_pred):
                idx = np.where(self.classes_ == y)[0][0]
                proba[i, idx] = 1.0

        if self.llm_ is None or self.committee_ is None or self.threshold_ is None:
            return proba

        # 对边界样本，用 LLM 的离散预测改写为接近 one-hot 的概率
        uncert = self._compute_uncertainty(X)
        boundary_mask = uncert >= self.threshold_

        if boundary_mask.any():
            X_boundary = X[boundary_mask]
            llm_pred = self.llm_.predict(X_boundary)
            for i, y in zip(np.where(boundary_mask)[0], llm_pred):
                # 把 LLM 预测转成 [0.001,...,0.999,...] 的 one-hot 近似
                idx = np.where(self.classes_ == y)[0][0]
                row = np.full(self.n_classes_, 0.001, dtype=float)
                row[idx] = 0.999
                proba[i] = row

        return proba

warnings.filterwarnings('ignore')

# Ensure output directories exist
os.makedirs('./analysis', exist_ok=True)
os.makedirs('./models', exist_ok=True)  # Model results go here

# Load processed data
print("=" * 80)
print("CHILD APPENDICITIS PREDICTION - MODEL TRAINING")
print("=" * 80)

print("\nLoading processed data...")

X_train_no_us = np.load('./processed_data/X_train_no_ultrasound.npy',
                        allow_pickle=True)
X_test_no_us = np.load('./processed_data/X_test_no_ultrasound.npy',
                       allow_pickle=True)
y_train_no_us = np.load('./processed_data/y_train_no_ultrasound.npy',
                        allow_pickle=True)
y_test_no_us = np.load('./processed_data/y_test_no_ultrasound.npy',
                       allow_pickle=True)

X_no_us = np.load('./processed_data/X_no_ultrasound.npy', allow_pickle=True)
y_no_us = np.load('./processed_data/y_no_ultrasound.npy', allow_pickle=True)

feature_names_no_us = np.load(
    './processed_data/feature_names_no_ultrasound.npy', allow_pickle=True)
target_names = np.load('./processed_data/target_names.npy', allow_pickle=True)
target_meta = np.load(
    './processed_data/target_meta.npy', allow_pickle=True
).item()

print(f"Training data (no ultrasound): {X_train_no_us.shape}")
print(f"Test data (no ultrasound): {X_test_no_us.shape}")
print(f"Targets: {target_names}")

# Report missing values
print(
    f"Missing values in training data (no ultrasound): {np.isnan(X_train_no_us).sum()}"
)
print(
    f"Missing values in test data (no ultrasound): {np.isnan(X_test_no_us).sum()}"
)

# 读取变量说明：Variable Name in Data Files / Explanation
meta_df = pd.read_excel("app_data.xlsx")

# 容错一下列名（防止大小写或空格差异）
col_var = None
col_exp = None
for c in meta_df.columns:
    if "variable" in str(c).lower() and "data" in str(c).lower():
        col_var = c
    if "explanation" in str(c).lower():
        col_exp = c

if col_var is None or col_exp is None:
    raise ValueError(
        f"Cannot find 'Variable Name in Data Files' / 'Explanation' columns in app_data.xlsx, got columns={meta_df.columns}"
    )

feature_explanations = {}
for _, row in meta_df.iterrows():
    var_name = str(row[col_var]).strip()
    exp = str(row[col_exp]).strip()
    if var_name and var_name != "nan":
        feature_explanations[var_name] = exp

print(f"\nLoaded {len(feature_explanations)} feature explanations from app_data.xlsx")

judge = False

if judge:
    # ============================================================
    # 读入 LLM 三个风险特征，并与原始特征拼接
    # 约定列顺序: 0=Diagnosis, 1=Severity, 2=Management
    # ============================================================
    llm_risk_train_path = "./processed_data/ds_risk_train_no_ultrasound.npy"
    llm_risk_test_path  = "./processed_data/ds_risk_test_no_ultrasound.npy"

    if os.path.exists(llm_risk_train_path) and os.path.exists(llm_risk_test_path):
        print("\nLoading LLM risk features and concatenating with original X ...")

        llm_risk_train = np.load(llm_risk_train_path, allow_pickle=True)
        llm_risk_test = np.load(llm_risk_test_path, allow_pickle=True)

        print(f"LLM risk train shape: {llm_risk_train.shape}")
        print(f"LLM risk test  shape: {llm_risk_test.shape}")

        # 简单 sanity check
        assert llm_risk_train.shape[0] == X_train_no_us.shape[0]
        assert llm_risk_test.shape[0] == X_test_no_us.shape[0]
        assert llm_risk_train.shape[1] == 3
        assert llm_risk_test.shape[1] == 3

        # 在列维度上拼接: [原始特征, LLM 风险]
        X_train_no_us = np.hstack([X_train_no_us, llm_risk_train])
        X_test_no_us  = np.hstack([X_test_no_us,  llm_risk_test])

        # 更新特征名
        llm_feature_names = np.array([
            "LLM_risk_Diagnosis",    # LLM 认为“有阑尾炎”的风险
            "LLM_risk_Severity",     # LLM 认为“复杂/严重阑尾炎”的风险
            "LLM_risk_Management",   # LLM 认为“需要手术/侵入性干预”的风险
        ], dtype=object)

        feature_names_no_us = np.concatenate(
            [feature_names_no_us, llm_feature_names]
        )

        # print(f"Original feature dim: {X_train_no_us.shape[1]}")
        # print(f"New feature dim      : {X_train_with_llm.shape[1]}")
    else:
        print("\n[Warning] LLM risk feature files not found, "
            "will only use original features.")
        X_train_with_llm = None
        X_test_with_llm = None
        feature_names_with_llm = None

def get_models_with_imputation(feature_names, target_name, id2label=None):
    """
    Ensemble tree models + LLM + 一些“新型树模型近似实现”.
    All models follow the same data handling strategy as the original code.
    """

    imputer = SimpleImputer(strategy='median')

    # --------------------------------------------------
    # 1. 基础树模型
    # --------------------------------------------------
    models = {
        # 1) Random Forest
        'Random Forest': Pipeline([
            ('imputer', imputer),
            ('clf', RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            ))
        ]),

        # 2) Gradient Boosting (GBDT)
        'Gradient Boosting': Pipeline([
            ('imputer', imputer),
            ('clf', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ))
        ]),

        # 3) AdaBoost
        'AdaBoost': Pipeline([
            ('imputer', imputer),
            ('clf', AdaBoostClassifier(
                n_estimators=300,
                learning_rate=0.05,
                random_state=42,
            ))
        ]),

        # 4) LightGBM
        'LightGBM': Pipeline([
            ('imputer', imputer),
            ('clf', LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ))
        ]),

        # 5) CatBoost
        'CatBoost': Pipeline([
            ('imputer', imputer),
            ('clf', CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                loss_function='MultiClass',
                random_seed=42,
                verbose=False,
            ))
        ]),
    }

    # --------------------------------------------------
    # 2. Neural / Diff Trees 类：Neural Decision Forest
    # --------------------------------------------------
    models['Neural Decision Forest'] = Pipeline([
        ('imputer', imputer),
        ('clf', NeuralDecisionForest(
            hidden_dim=64,
            n_estimators=200,
            max_depth=None,
            batch_size=32,
            epochs=20,
            lr=1e-3,
            device=None,        # None = 自动选 cuda / cpu
            random_state=42,
        ))
    ])

    # --------------------------------------------------
    # 3. DGBF: 用 XGBoost 近似 Distributed GBDT
    # --------------------------------------------------
    # models['DGBF (XGBoost approx)'] = Pipeline([
    #     ('imputer', imputer),
    #     ('clf', XGBClassifier(
    #         n_estimators=400,
    #         learning_rate=0.05,
    #         max_depth=5,
    #         subsample=0.8,
    #         colsample_bytree=0.8,
    #         objective='multi:softprob',  # 兼容二分类 & 多分类
    #         eval_metric='mlogloss',
    #         tree_method='hist',          # 直方图 / 分布式友好
    #         random_state=42,
    #         n_jobs=-1,
    #     ))
    # ])

    # --------------------------------------------------
    # 4. GRANDE: 用 HistGradientBoostingClassifier 近似
    # --------------------------------------------------
    models['GRANDE (HistGBDT approx)'] = Pipeline([
        ('imputer', imputer),
        ('clf', HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=None,
            random_state=42,
        ))
    ])

    # --------------------------------------------------
    # 5. DT-GFN: 用 ExtraTrees 做“探索性”树模型近似
    # --------------------------------------------------
    models['DT-GFN (ExtraTrees approx)'] = Pipeline([
        ('imputer', imputer),
        ('clf', ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            bootstrap=False,
        ))
    ])

    # --------------------------------------------------
    # 6. MorphBoost: 两个不同深度的 GBDT 做 soft voting
    # --------------------------------------------------
    gb_stage1 = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )
    gb_stage2 = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    models['MorphBoost (stacked GBDT)'] = Pipeline([
        ('imputer', imputer),
        ('clf', VotingClassifier(
            estimators=[('gb1', gb_stage1), ('gb2', gb_stage2)],
            voting='soft'
        ))
    ])

    # --------------------------------------------------
    # 7. Meta-Tree Boosting: RF + GBDT + LGBM 的 meta-ensemble
    # --------------------------------------------------
    meta_estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )),
        ('lgbm', LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )),
    ]
    models['Meta-Tree Boosting'] = Pipeline([
        ('imputer', imputer),
        ('clf', VotingClassifier(
            estimators=meta_estimators,
            voting='soft'
        ))
    ])
    # --------------------------------------------------
    # 9. TEL: Tree Ensemble Layer (trees as a high-level layer)
    # --------------------------------------------------
    models['TEL (Tree Ensemble Layer)'] = Pipeline([
        ('imputer', imputer),
        ('clf',
         TreeEnsembleLayerClassifier(
             rf_n_estimators=200,
             rf_max_depth=None,
             gb_n_estimators=200,
             gb_learning_rate=0.05,
             gb_max_depth=3,
             C=1.0,
             random_state=42,
         ))
    ])

    # --------------------------------------------------
    # 10. TreeNet: stacked tree ensembles
    # --------------------------------------------------
    models['TreeNet'] = Pipeline([
        ('imputer', imputer),
        ('clf',
         TreeNetClassifier(
             level1_n_estimators=200,
             level1_max_depth=None,
             level2_n_estimators=200,
             level2_max_depth=None,
             random_state=42,
         ))
    ])
    
    # --------------------------------------------------
    # 11. Boundary LLM Hybrid：
    #     committee 选边界样本 -> 边界交给 LLM，其他交给 base tree
    # --------------------------------------------------
    boundary_llm = BoundaryLLMHybrid(
        base_estimator=TreeNetClassifier(
             level1_n_estimators=200,
             level1_max_depth=None,
             level2_n_estimators=200,
             level2_max_depth=None,
             random_state=42,
         ),
        boundary_quantile=0.01,  # top 20% 最不确定的样本交给 LLM
        llm_estimator=LLMClassifier(
            llm=llm_gpt,
            feature_names=feature_names,
            target_name=target_name,
            id2label=id2label,
            # few-shot 配置
            n_shots_per_class=4,
            max_total_shots=8,
            # 错题迭代配置
            refine_rounds=1,                 # 在训练集上迭代 1 轮
            refine_max_errors_per_round=32,  # 每轮最多用 32 个错分样本来更新 few-shot
            role_description=(
                "You are a senior pediatric emergency surgeon. "
                "You will see clinical features of one child and must predict "
                f"the label for the target: \"{target_name}\"."
            ),
            feature_explanations=feature_explanations,
        ),
        random_state=42,
    )

    models['Boundary ds Hybrid'] = Pipeline([
        ('imputer', imputer),
        ('clf', boundary_llm),
    ])

    # --------------------------------------------------
    # 8. LLM Doctor（保持不变）
    # --------------------------------------------------
    gb_stage1 = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    )
    gb_stage2 = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    models['ds simple'] = LLMClassifier(
        llm=llm_ds,
        feature_names=feature_names,
        target_name=target_name,
        id2label=id2label,
        # few-shot 配置
        n_shots_per_class=0,
        max_total_shots=0,
        # 错题迭代配置
        refine_rounds=0,                 # 在训练集上迭代 1 轮
        refine_max_errors_per_round=32,  # 每轮最多用 32 个错分样本来更新 few-shot
        role_description=(
            "You are a senior pediatric emergency surgeon. "
            "You will see clinical features of one child and must predict "
            f"the label for the target: \"{target_name}\"."
        ),
        feature_explanations=feature_explanations,
    )
    models['ds with examples'] = LLMClassifier(
        llm=llm_ds,
        feature_names=feature_names,
        target_name=target_name,
        id2label=id2label,
        # few-shot 配置
        n_shots_per_class=4,
        max_total_shots=8,
        # 错题迭代配置
        refine_rounds=0,                 # 在训练集上迭代 1 轮
        refine_max_errors_per_round=32,  # 每轮最多用 32 个错分样本来更新 few-shot
        role_description=(
            "You are a senior pediatric emergency surgeon. "
            "You will see clinical features of one child and must predict "
            f"the label for the target: \"{target_name}\"."
        ),
        feature_explanations=feature_explanations,
    )
    models['ds with examples & refine'] = LLMClassifier(
        llm=llm_ds,
        feature_names=feature_names,
        target_name=target_name,
        id2label=id2label,
        # few-shot 配置
        n_shots_per_class=4,
        max_total_shots=8,
        # 错题迭代配置
        refine_rounds=1,                 # 在训练集上迭代 1 轮
        refine_max_errors_per_round=32,  # 每轮最多用 32 个错分样本来更新 few-shot
        role_description=(
            "You are a senior pediatric emergency surgeon. "
            "You will see clinical features of one child and must predict "
            f"the label for the target: \"{target_name}\"."
        ),
        feature_explanations=feature_explanations,
    )
    models['ds with examples & refine & base'] = LLMClassifier(
        llm=llm_ds,
        feature_names=feature_names,
        target_name=target_name,
        id2label=id2label,
        # few-shot 配置
        n_shots_per_class=4,
        max_total_shots=8,
        # 错题迭代配置
        refine_rounds=1,                 # 在训练集上迭代 1 轮
        refine_max_errors_per_round=32,  # 每轮最多用 32 个错分样本来更新 few-shot
        role_description=(
            "You are a senior pediatric emergency surgeon. "
            "You will see clinical features of one child and must predict "
            f"the label for the target: \"{target_name}\"."
        ),
        base_estimators={
            "HistGBDT": HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, random_state=42
            ),
            "TreeNet": TreeNetClassifier(
                level1_n_estimators=200,
                level1_max_depth=None,
                level2_n_estimators=200,
                level2_max_depth=None,
                random_state=42,
            ),
            "MorphBoost": VotingClassifier(
                estimators=[('gb1', gb_stage1), ('gb2', gb_stage2)],
                voting='soft'
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
            "NeuralDecisionForest":  NeuralDecisionForest(
                hidden_dim=64,
                n_estimators=200,
                max_depth=None,
                batch_size=32,
                epochs=20,
                lr=1e-3,
                device=None,        # None = 自动选 cuda / cpu
                random_state=42,
            ),
            "DT-GFN": ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
                bootstrap=False,
            )
        },
        feature_explanations=feature_explanations,
    )
    models['gpt simple'] = LLMClassifier(
        llm=llm_gpt,
        feature_names=feature_names,
        target_name=target_name,
        id2label=id2label,
        # few-shot 配置
        n_shots_per_class=0,
        max_total_shots=0,
        # 错题迭代配置
        refine_rounds=0,                 # 在训练集上迭代 1 轮
        refine_max_errors_per_round=32,  # 每轮最多用 32 个错分样本来更新 few-shot
        role_description=(
            "You are a senior pediatric emergency surgeon. "
            "You will see clinical features of one child and must predict "
            f"the label for the target: \"{target_name}\"."
        ),
        feature_explanations=feature_explanations,
    )
    models['gpt with examples'] = LLMClassifier(
        llm=llm_gpt,
        feature_names=feature_names,
        target_name=target_name,
        id2label=id2label,
        # few-shot 配置
        n_shots_per_class=4,
        max_total_shots=8,
        # 错题迭代配置
        refine_rounds=0,                 # 在训练集上迭代 1 轮
        refine_max_errors_per_round=32,  # 每轮最多用 32 个错分样本来更新 few-shot
        role_description=(
            "You are a senior pediatric emergency surgeon. "
            "You will see clinical features of one child and must predict "
            f"the label for the target: \"{target_name}\"."
        ),
        feature_explanations=feature_explanations,
    )
    models['gpt with examples & refine'] = LLMClassifier(
        llm=llm_gpt,
        feature_names=feature_names,
        target_name=target_name,
        id2label=id2label,
        # few-shot 配置
        n_shots_per_class=4,
        max_total_shots=8,
        # 错题迭代配置
        refine_rounds=1,                 # 在训练集上迭代 1 轮
        refine_max_errors_per_round=32,  # 每轮最多用 32 个错分样本来更新 few-shot
        role_description=(
            "You are a senior pediatric emergency surgeon. "
            "You will see clinical features of one child and must predict "
            f"the label for the target: \"{target_name}\"."
        ),
        feature_explanations=feature_explanations,
    )
    models['gpt with examples & refine & base'] = LLMClassifier(
        llm=llm_gpt,
        feature_names=feature_names,
        target_name=target_name,
        id2label=id2label,
        # few-shot 配置
        n_shots_per_class=4,
        max_total_shots=8,
        # 错题迭代配置
        refine_rounds=1,                 # 在训练集上迭代 1 轮
        refine_max_errors_per_round=32,  # 每轮最多用 32 个错分样本来更新 few-shot
        role_description=(
            "You are a senior pediatric emergency surgeon. "
            "You will see clinical features of one child and must predict "
            f"the label for the target: \"{target_name}\"."
        ),
        base_estimators={
            "HistGBDT": HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, random_state=42
            ),
            "TreeNet": TreeNetClassifier(
                level1_n_estimators=200,
                level1_max_depth=None,
                level2_n_estimators=200,
                level2_max_depth=None,
                random_state=42,
            ),
            "MorphBoost": VotingClassifier(
                estimators=[('gb1', gb_stage1), ('gb2', gb_stage2)],
                voting='soft'
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
            "NeuralDecisionForest":  NeuralDecisionForest(
                hidden_dim=64,
                n_estimators=200,
                max_depth=None,
                batch_size=32,
                epochs=20,
                lr=1e-3,
                device=None,        # None = 自动选 cuda / cpu
                random_state=42,
            ),
            "DT-GFN": ExtraTreesClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
                bootstrap=False,
            )
        },
        feature_explanations=feature_explanations,
    )

    return models




def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a single model"""
    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    # y_prob = None
    # if hasattr(model, 'predict_proba'):
    #     y_prob = model.predict_proba(X_test)

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
    # auc = None
    # if y_prob is not None:
    #     try:
    #         if n_classes == 2:
    #             auc = roc_auc_score(y_test, y_prob[:, 1])
    #         else:
    #             auc = roc_auc_score(y_test,
    #                                 y_prob,
    #                                 multi_class='ovr',
    #                                 average='weighted')
    #     except:
    #         auc = None

    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        # 'auc_roc': auc,
        'y_pred': y_pred,
        # 'y_prob': y_prob
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

    tname = str(target_name)
    if tname in target_meta:
        encoding = target_meta[tname]["encoding"]
        id2label = {v: k for k, v in encoding.items()}
    else:
        # 不在 meta 里就退化为数字类本身
        id2label = {}

    results = {}
    models = get_models_with_imputation(
        feature_names=feature_names,
        target_name=target_name,
        id2label=id2label,
    )

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        result = evaluate_model(model, X_train, X_test, y_train, y_test,
                                model_name)
        results[model_name] = result

        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1-Score:  {result['f1_score']:.4f}")
        # if result['auc_roc'] is not None:
        #     print(f"  AUC-ROC:   {result['auc_roc']:.4f}")

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


# def train_with_selected_features(X_train, X_test, y_train, y_test,
#                                  feature_names, selected_features,
#                                  target_name):
#     """Train models using only selected features with pre-split data"""
#     print(f"\n{'='*60}")
#     print(f"TRAINING WITH SELECTED FEATURES for {target_name}")
#     print(f"Using {len(selected_features)} features")
#     print(f"{'='*60}")

#     # Handle rare classes
#     unique_train, counts_train = np.unique(y_train, return_counts=True)
#     valid_classes = unique_train[counts_train >= 5]
#     if len(valid_classes) < len(unique_train):
#         train_mask = np.isin(y_train, valid_classes)
#         X_train = X_train[train_mask]
#         y_train = y_train[train_mask]
#         test_mask = np.isin(y_test, valid_classes)
#         X_test = X_test[test_mask]
#         y_test = y_test[test_mask]
#         label_map = {
#             old: new
#             for new, old in enumerate(sorted(np.unique(y_train)))
#         }
#         y_train = np.array([label_map[v] for v in y_train])
#         y_test = np.array([label_map.get(v, -1) for v in y_test])
#         valid_test = y_test >= 0
#         X_test = X_test[valid_test]
#         y_test = y_test[valid_test]

#     # Get indices of selected features
#     feature_idx = [
#         i for i, f in enumerate(feature_names) if f in selected_features
#     ]
#     X_train_selected = X_train[:, feature_idx]
#     X_test_selected = X_test[:, feature_idx]

#     results = {}
#     models = get_models_with_imputation()

#     for model_name, model in models.items():
#         print(f"\nTraining {model_name}...")
#         result = evaluate_model(model, X_train_selected, X_test_selected,
#                                 y_train, y_test, model_name)
#         results[model_name] = result

#         print(f"  Accuracy:  {result['accuracy']:.4f}")
#         print(f"  F1-Score:  {result['f1_score']:.4f}")
#         if result['auc_roc'] is not None:
#             print(f"  AUC-ROC:   {result['auc_roc']:.4f}")

#     return results


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
def plot_prediction_correlation(results, target_name, save_path):
    """
    根据各个模型在同一个测试集上的预测标签 y_pred，
    计算模型之间预测的一致性（皮尔逊相关系数），并画成热图。

    参数
    ----
    results : dict
        形如 {model_name: {"y_pred": ..., ...}, ...}，
        就是 train_and_evaluate_all_models 返回的那个 results_all。
    target_name : str
        当前任务名，用在标题里。
    save_path : str
        保存图片的路径。
    """
    preds_dict = {}
    n_samples = None

    for model_name, metrics in results.items():
        y_pred = metrics.get("y_pred", None)
        if y_pred is None:
            # 这个模型没有预测结果，就跳过
            continue

        arr = np.asarray(y_pred)

        # 如果是二维 (n, 1) 或 (1, n)，压成一维
        if arr.ndim > 1:
            arr = arr.reshape(arr.shape[0], -1)[:, 0]

        # 对齐长度（如果有模型长度不一致，直接丢掉）
        if n_samples is None:
            n_samples = len(arr)
        else:
            if len(arr) != n_samples:
                print(f"[Warning] model {model_name} has different length "
                      f"({len(arr)} vs {n_samples}), skipped in correlation.")
                continue

        preds_dict[model_name] = arr

    # 至少要有两个模型才能算相关性
    if len(preds_dict) < 2:
        print("[Info] Less than 2 models with valid predictions, "
              "skip correlation heatmap.")
        return

    # 构造 DataFrame: shape (n_samples, n_models)
    pred_df = pd.DataFrame(preds_dict)

    # 计算皮尔逊相关系数矩阵
    corr = pred_df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        square=True
    )
    plt.title(f"Prediction Correlation - {target_name}")
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
    y_train = y_train_no_us[:, target_idx]
    y_test = y_test_no_us[:, target_idx]

    # Train with all features using pre-split data
    results_all, X_tr, X_te, y_tr, y_te = train_and_evaluate_all_models(
        X_train_no_us.copy(), X_test_no_us.copy(), y_train.copy(), y_test.copy(),
        target_name, feature_names_no_us, "All Features")
    all_results[(target_name, 'All Features')] = results_all

    # Plot confusion matrices
    plot_confusion_matrices(
        results_all, y_te, target_name, "All Features",
        f'./analysis/confusion_matrix_{target_name}_all.png')
    
    # Plot confusion matrices
    plot_confusion_matrices(
        results_all, y_te, target_name, "All Features",
        f'./analysis/confusion_matrix_{target_name}_all.png')

    # Plot prediction correlation heatmap between models
    plot_prediction_correlation(
        results_all,
        target_name,
        f'./analysis/pred_corr_{target_name}_all.png'
    )

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
    print(f"  Precision: {best_row['Precision']:.4f}")
    print(f"  Recall: {best_row['Recall']:.4f}")
    print(f"  F1-Score: {best_row['F1-Score']:.4f}")
    # if pd.notna(best_row['AUC-ROC']):
    #     print(f"  AUC-ROC:  {best_row['AUC-ROC']:.4f}")

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
