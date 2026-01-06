import numpy as np
X_train_no_us = np.load('./processed_data/X_train_no_ultrasound.npy',
                        allow_pickle=True)
X_test_no_us = np.load('./processed_data/X_test_no_ultrasound.npy',
                       allow_pickle=True)
y_train_no_us = np.load('./processed_data/y_train_no_ultrasound.npy',
                        allow_pickle=True)
y_test_no_us = np.load('./processed_data/y_test_no_ultrasound.npy',
                       allow_pickle=True)


class LLMTripleRiskScorer:
    """
    一次性为每个病例生成三个风险评分：
      - p_diagnosis   : 有阑尾炎的风险（0~1）
      - p_severity    : 复杂/严重阑尾炎风险（0~1）
      - p_management  : 需要手术/侵入性干预的风险（0~1）

    使用思路：
      1) fit(X_train, Y_train) 只用训练集抽 few-shot 例子
      2) transform(X)           对任意 X 输出 (n_samples, 3) 的风险矩阵
    """

    def __init__(
        self,
        llm,
        feature_names,
        target_names,
        target_meta,
        n_shots_per_class=3,
        max_total_shots=12,
        feature_explanations=None,   # NEW: 加入变量说明
    ):
        self.llm = llm
        self.feature_names = list(feature_names)
        self.target_names = [str(t) for t in target_names]
        self.target_meta = target_meta
        self.n_shots_per_class = n_shots_per_class
        self.max_total_shots = max_total_shots

        # NEW: 保存 “Variable Name in Data Files -> Explanation”
        self.feature_explanations = feature_explanations or {}

        # few-shot + label 映射
        self.few_shot_examples = []   # 每个元素: {"x": ..., "labels": np.array([...])}
        self.id2label_per_target = {}
        self.diag_idx = None
        self.severity_idx = None
        self.management_idx = None

        # 预先解析每个 target 的 id2label
        for tname in self.target_names:
            meta = self.target_meta.get(tname, {})
            encoding = meta.get("encoding", {})
            id2label = {int(v): str(k) for k, v in encoding.items()}
            self.id2label_per_target[tname] = id2label

        # 找到三个任务在 target_names 里的位置
        lowers = [t.lower() for t in self.target_names]
        # Diagnosis
        for i, s in enumerate(lowers):
            if "diag" in s:
                self.diag_idx = i
                break
        # Severity
        for i, s in enumerate(lowers):
            if "sever" in s:
                self.severity_idx = i
                break
        # Management
        for i, s in enumerate(lowers):
            if "manage" in s:
                self.management_idx = i
                break

        if self.diag_idx is None or self.severity_idx is None or self.management_idx is None:
            raise ValueError(
                f"Cannot locate Diagnosis/Severity/Management in target_names={self.target_names}"
            )

    # -----------------------------
    # few-shot 构造（基于 Diagnosis 类中心）
    # -----------------------------
    def fit(self, X_train, Y_train):
        """
        X_train: (n_train, n_features)
        Y_train: (n_train, n_targets)  这里的列顺序需要和 target_names 对齐
        """
        X = np.asarray(X_train, dtype=float)
        Y = np.asarray(Y_train)

        # 用 Diagnosis 这列来分层选例子
        y_diag = Y[:, self.diag_idx]
        classes = np.unique(y_diag)

        if X.size == 0:
            return self

        # 先对 X 做简单均值填补，方便算“类中心”
        X_imp = X.copy()
        col_means = np.nanmean(X_imp, axis=0)
        inds = np.where(np.isnan(X_imp))
        X_imp[inds] = np.take(col_means, inds[1])

        examples = []
        for cls in classes:
            idx = np.where(y_diag == cls)[0]
            if len(idx) == 0:
                continue
            X_c = X_imp[idx]
            center = X_c.mean(axis=0)
            dists = np.linalg.norm(X_c - center, axis=1)

            k = min(self.n_shots_per_class, len(idx))
            chosen_local = np.argsort(dists)[:k]
            chosen_idx = idx[chosen_local]

            for j in chosen_idx:
                examples.append({
                    "x": X[j].copy(),
                    "labels": Y[j].copy(),  # 一次性带上三个 target 的标签
                })

        # 截断 few-shot 总数
        if len(examples) > self.max_total_shots:
            examples = examples[:self.max_total_shots]

        self.few_shot_examples = examples
        return self

    # -----------------------------
    # 文本化一个病例（加上 Explanation）
    # -----------------------------
    def _case_to_text(self, x):
        parts = []
        for name, val in zip(self.feature_names, x):
            # NEW: 查找该变量的 Explanation
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

    # -----------------------------
    # 三个任务的自然语言描述
    # -----------------------------
    def _risk_sentences(self):
        s_diag = (
            "Diagnosis: whether the child truly has acute appendicitis."
        )
        s_sev = (
            "Severity: whether the appendicitis is complicated or severe "
            "(for example perforation, abscess, or generalized peritonitis)."
        )
        s_mng = (
            "Management: whether the child requires surgical management "
            "(appendectomy or another invasive operative procedure)."
        )
        return s_diag, s_sev, s_mng

    # -----------------------------
    # 构造 prompt（一次性问三个 0/1 判断）
    # -----------------------------
    def _build_prompt(self, x):
        case_text = self._case_to_text(x)
        s_diag, s_sev, s_mng = self._risk_sentences()

        # 每个 target 的整数编码 → 文本（只是辅助解释，可保留）
        mapping_lines = []
        for tname in self.target_names:
            id2label = self.id2label_per_target.get(tname, {})
            if not id2label:
                continue
            mapping_lines.append(f"{tname}:")
            for lid, lname in sorted(id2label.items(), key=lambda z: z[0]):
                mapping_lines.append(f"  {lid} = {lname}")
        mapping_text = "\n".join(mapping_lines)

        # few-shot 例子
        examples_blocks = []
        for i, ex in enumerate(self.few_shot_examples):
            feat_text = self._case_to_text(ex["x"])
            labels = ex["labels"]
            lines = [f"Example {i+1}:"]
            lines.append(f"Features: {feat_text}")
            for t_idx, tname in enumerate(self.target_names):
                lid = int(labels[t_idx])
                lname = self.id2label_per_target.get(tname, {}).get(
                    lid, f"class_{lid}"
                )
                lines.append(f"{tname} label: {lid} ({lname})")
            examples_blocks.append("\n".join(lines))
        examples_text = ""
        if examples_blocks:
            examples_text = (
                "Here are some example children with their features and true labels "
                "for each target:\n\n" + "\n\n".join(examples_blocks) + "\n"
            )

        prompt = f"""
You are a senior pediatric emergency surgeon.

You will see the clinical features of one child with suspected appendicitis.
There are three prediction targets:

1) {s_diag}
2) {s_sev}
3) {s_mng}

For each target, you must make a binary decision:
- 1 = the more serious / positive outcome is present
- 0 = the more serious / positive outcome is NOT present

The dataset internally encodes labels as integer codes. Here is the mapping:
{mapping_text}

{examples_text}
Now here is a NEW child (different from all examples above).

The features of this child are:
{case_text}

Based only on these features and your clinical knowledge (and not on the true labels),
decide three binary labels:

- y_diagnosis   = 1 if the child truly has appendicitis, else 0.
- y_severity    = 1 if the child has complicated/severe appendicitis, else 0.
- y_management  = 1 if the child requires surgical / operative management, else 0.

Output your answer on a single line in the following exact format:
y_diagnosis=AA; y_severity=BB; y_management=CC

where AA, BB and CC are integers, each either 0 or 1.
Do not output any extra words or explanation.
"""
        return prompt

    # -----------------------------
    # 解析 LLM 输出的三个 0/1 整数
    # -----------------------------
    def _parse_scores(self, raw_text):
        """
        解析形如：
          y_diagnosis=1; y_severity=0; y_management=1
        的输出，返回 np.array([1,0,1]) （float 或 int 都可）
        """
        # 默认全 0，当 LLM 出错时兜底
        if raw_text is None:
            return np.array([0.0, 0.0, 0.0], dtype=float)

        text = str(raw_text)

        def _get01(pattern):
            m = re.search(pattern, text)
            if m:
                try:
                    v = int(m.group(1))
                except ValueError:
                    v = 0
            else:
                v = 0
            # 强制裁剪到 {0,1}
            v = 1 if v >= 1 else 0
            return float(v)

        y_diag = _get01(r"y_diagnosis\s*=\s*(\d+)")
        y_sev  = _get01(r"y_severity\s*=\s*(\d+)")
        y_mng  = _get01(r"y_management\s*=\s*(\d+)")
        return np.array([y_diag, y_sev, y_mng], dtype=float)

    # -----------------------------
    # 对一批样本生成 (n_samples, 3) 0/1 特征
    # -----------------------------
    def transform(self, X, max_workers=8):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        scores = np.zeros((n, 3), dtype=float)

        def _one(i):
            prompt = self._build_prompt(X[i])
            raw = self.llm.query(prompt)
            return self._parse_scores(raw)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_one, i): i for i in range(n)}
            for f in tqdm(as_completed(futures),
                          total=n,
                          desc="LLM triple labels (Diagnosis/Severity/Management)"):
                i = futures[f]
                try:
                    scores[i] = f.result()
                except Exception:
                    # 出错就全 0
                    scores[i] = np.array([0.0, 0.0, 0.0], dtype=float)

        return scores 
def build_llm_risk_features():
    """
    使用 LLM 在【训练集 + 测试集】上，一次性生成三个风险特征：
      - col 0: Diagnosis risk   (有阑尾炎的风险)
      - col 1: Severity risk    (复杂/严重阑尾炎风险)
      - col 2: Management risk  (需要手术/侵入性干预的风险)

    输出:
      ./processed_data/llm_risk_train_no_ultrasound.npy  (n_train, 3)
      ./processed_data/llm_risk_test_no_ultrasound.npy   (n_test, 3)
    """
    print("\n" + "=" * 80)
    print("BUILDING LLM-BASED RISK FEATURES (Diagnosis / Severity / Management)")
    print("=" * 80)

    # ---- 这里直接复用你脚本里已经加载好的全局变量 ----
    global X_train_no_us, X_test_no_us, y_train_no_us, y_test_no_us
    global feature_names_no_us, target_names, target_meta, llm

    n_train = X_train_no_us.shape[0]
    n_test = X_test_no_us.shape[0]

    # risk 矩阵，列顺序固定为 [Diagnosis, Severity, Management]
    llm_risk_train = np.zeros((n_train, 3), dtype=np.float32)
    llm_risk_test = np.zeros((n_test, 3), dtype=np.float32)

    # 构建打分器，只在 fit 里用训练集抽 few-shot
    scorer = LLMTripleRiskScorer(
        llm=llm,
        feature_names=feature_names_no_us,
        target_names=target_names,
        target_meta=target_meta,
        n_shots_per_class=3,
        max_total_shots=12,
        feature_explanations=feature_explanations,  # NEW
    )

    print("\n[LLM risk] fitting few-shot examples on TRAIN set ...")
    scorer.fit(X_train_no_us, y_train_no_us)

    print("\n[LLM risk] scoring TRAIN set ...")
    llm_risk_train[:, :] = scorer.transform(X_train_no_us)

    print("\n[LLM risk] scoring TEST set ...")
    llm_risk_test[:, :] = scorer.transform(X_test_no_us)

    out_train = "./processed_data/gpt_risk_train_no_ultrasound.npy"
    out_test = "./processed_data/gpt_risk_test_no_ultrasound.npy"
    np.save(out_train, llm_risk_train)
    np.save(out_test, llm_risk_test)

    print("\nSaved LLM risk features:")
    print(f"  - Train: {out_train}, shape = {llm_risk_train.shape}")
    print(f"  - Test : {out_test}, shape = {llm_risk_test.shape}")
    print("\n" + "=" * 80)
    print("LLM RISK FEATURE GENERATION COMPLETED")
    print("=" * 80)
    
if __name__ == "__main__":
    build_llm_risk_features()