Here is the Markdown representation of the presentation slides.

# Statistical Learning Based Pediatric Appendicitis Prediction Project

**Authors:** 苑闻, 周景星, 王梓桐
**Date:** 2025.12.23

---

## Part 1: Problem and Data Overview

---

### Problem and Data Overview

*   **Pediatric appendicitis (小儿阑尾炎)** is one of the most common acute surgical emergencies in childhood.
*   **NIH** reported a lifetime risk of 8.6% for males and 6.7% for females.
*   Early and accurate diagnosis can prevent complications.
*   We adopt the dataset provided by **UCI**.
    *   Sourced from Children's Hospital St. Hedwig, Regensburg, Germany.
    *   Collected between 2016-2021.
    *   782 patients, with a total of 58 features.

**Feature Categories:**

| Category                 | Count | Examples                                                             |
| :----------------------- | :---- | :------------------------------------------------------------------- |
| Demographic Features     | 6     | Age, Sex, Height, Weight, BMI, Length_of_Stay                        |
| Clinical Scoring Systems | 2     | Alvarado Score, Paediatric Appendicitis Score                        |
| Clinical Examination     | 10    | Migratory Pain, Lower Right Abdominal Pain, Nausea, Body Temperature |
| Peritonitis Indicators   | 3     | Generalized, Local, No peritonitis (one-hot encoded)                 |
| Stool Characteristics    | 4     | Constipation, Diarrhea, Normal (one-hot encoded)                     |
| Laboratory Tests         | 9     | WBC Count, Neutrophil Percentage, CRP, Hemoglobin                    |
| Urinalysis               | 12    | Ketones, RBC, WBC in urine (one-hot encoded)                         |

> **References:**
> *   Afridi MA, Khan I, Khalid MM, Ullah N. Combined clinical accuracy of inflammatory markers and ultrasound for the diagnosis of acute appendicitis. Ultrasound. 2023;31(4):266-272.
> *   Marcinkevičs, R., et.al. Regensburg Pediatric Appendicitis Dataset. 1.01, Zenodo, 2023.02.23

---

### Problem and Data Overview (Continued)

*   **Three target variables:** Diagnosis, Severity, Management.
*   The features contain both categorical and numerical data.
*   After preprocessing, we obtained **46 numerical features**, either continuous or binary.
*   We split the data into a **training set of size 624 (80%)** and a **test set of size 156 (20%)**.

![Distributions of Diagnosis, Management, and Severity](placeholder_image_distributions)

---

### Data Analysis

*   We first examine the correlation between different features, and between the target and the features.
*   The first group are basic physical indicators.
*   The second group are clinical scores and commonly used laboratory markers. Only numerical ones are visualized.
*   The last group are our targets.

![Correlation Heatmap including Age, BMI, Sex, Height, Weight, Clinical Scores, and Targets](placeholder_image_heatmap_1)

---

### Data Analysis (Correlations)

*   We first examine the correlation between different features, and between the target and the features.
*   We can see that the top correlation terms well align with prior knowledge.

**No Ultrasound: Top Correlations**
*   **Diagnosis:** Correlated with Segmented Neutrophils, Alvarado Score, Peritonitis_no, WBC Count.
*   **Management:** Correlated with Peritonitis_no, Peritonitis_local, Segmented Neutrophils.
*   **Severity:** Correlated with CRP, Segmented Neutrophils, WBC Count.

![Bar charts showing top 15 features correlated with Diagnosis, Management, and Severity](placeholder_image_bar_charts)
![Table showing feature correlations with target variables](placeholder_image_correlation_table)

---

### Data Analysis (Missing Data)

*   Of course, due to different clinical procedures and varying patient conditions, the features suffer from incompleteness.
*   Some features that align well with target variables also have large proportions of samples missing.
*   We will later focus on different methods handling missing features, thereby maximizing the utilization of already scarce data.

**Missing Data Summary:**

| Feature                                 | Missing Rate | Comment                             |
| :-------------------------------------- | :----------- | :---------------------------------- |
| Segmented_Neutrophils                   | 93.1%        | Laboratory, nearly complete missing |
| Urinalysis features (RBC, WBC, Ketones) | 25-26%       | Laboratory, moderate missingness    |
| Ipsilateral_Rebound_Tenderness          | 20.8%        | Clinical, examination gap           |
| Neutrophil_Percentage                   | 13.2%        | Laboratory                          |
| Clinical Scores (Alvarado, PAS)         | 6.6%         | Scoring systems                     |

---

## Part 2: Tree-Based Prediction Methods

---

### Basic Tree Method

*   **Random forest**
    *   Random Sampling with Replacement.
*   **Gradient Boosting**
    *   Predict residuals ($r_1 = y_1 - \hat{y}_1$, etc.).
*   **AdaBoost**
    *   Weighted dataset -> Strong learner.
*   **LightGBM**
    *   Leaf-wise tree growth.

![Diagrams of Random Forest, Gradient Boosting, AdaBoost, and LightGBM](placeholder_image_tree_methods_1)

---

### Basic Tree Method (Continued)

*   **CatBoost:** Oblivious Trees vs Regular Trees.
*   **Histogram-based Gradient Boosting:** Discretized features into histograms.
*   **Neural Decision Forest:** Data pre-processing -> Deep Neural Decision Forest.
*   **Tree Ensemble Layer:** Tree ensembles aggregation.

![Diagrams of CatBoost, Histogram Boosting, Neural Decision Forest, and Tree Ensemble Layer](placeholder_image_tree_methods_2)

---

### Basic Tree Method (Ensemble Strategies)

*   **TreeNet:**
    *   First train `rf1`, then use `(x, rf1)` to train `rf2`.
*   **MorphBoost:**
    *   Depth 2 and 4 GradientBoosting.
    *   Voting.
*   **Meta Tree Boosting:**
    *   RF, GB, LGBM vote.

---

### LLM: zero shot, small examples, refine, with base model, uncertainty

*   **Direct Ask LLM**
*   **Refine by give error predictions**
*   **Give several mean examples**
*   **Give base model results**
*   **Only solve difficult problem**

![Icons of robots illustrating LLM workflows](placeholder_image_llm_robots)

---

### LLM: provide new features

Based only on these features and your clinical knowledge (and not on the true labels), decide three binary labels:

*   `y_diagnosis` = 1 if the child truly has appendicitis, else 0.
*   `y_severity` = 1 if the child has complicated/severe appendicitis, else 0.
*   `y_management` = 1 if the child requires surgical / operative management, else 0.

---

### Results: Comparison of All Models (Confusion Matrices)

**(Diagnosis Target)**
*   **Models Shown:** Random Forest, Gradient Boosting, LightGBM, CatBoost, Neural Decision Forest, GRANDE (HistGBDT approx), DT-GFN (ExtraTrees approx), MorphBoost (stacked GBDT), Meta-Tree Boosting, TEL (Tree Ensemble Layer).

![Grid of confusion matrices for Diagnosis](placeholder_image_confusion_matrix_diagnosis_1)

---

### Results: Comparison of All Models (LLM/Hybrid Diagnosis)

**(Diagnosis Target - LLM/Hybrid)**
*   **Models Shown:** Boundary ds Hybrid, ds simple, ds with examples, ds with examples & refine, ds with examples & refine & base, gpt simple, gpt with examples, gpt with examples & refine, gpt with examples & refine & base.

![Grid of confusion matrices for Diagnosis using LLM methods](placeholder_image_confusion_matrix_diagnosis_2)

---

### Results: Comparison of All Models (Management)

**(Management Target)**
*   **Models Shown:** Random Forest, Gradient Boosting, LightGBM, CatBoost, Neural Decision Forest, GRANDE, DT-GFN, MorphBoost, Meta-Tree Boosting, TEL.

![Grid of confusion matrices for Management](placeholder_image_confusion_matrix_management_1)

---

### Results: Comparison of All Models (LLM/Hybrid Management)

**(Management Target - LLM/Hybrid)**
*   **Models Shown:** Boundary ds Hybrid, ds simple, ds with examples, ds with examples & refine, ds with examples & refine & base, gpt simple, gpt with examples, gpt with examples & refine, gpt with examples & refine & base.

![Grid of confusion matrices for Management using LLM methods](placeholder_image_confusion_matrix_management_2)

---

### Results: Comparison of All Models (Severity)

**(Severity Target)**
*   **Models Shown:** Random Forest, Gradient Boosting, LightGBM, CatBoost, Neural Decision Forest, GRANDE, DT-GFN, MorphBoost, Meta-Tree Boosting, TEL.

![Grid of confusion matrices for Severity](placeholder_image_confusion_matrix_severity_1)

---

### Results: Comparison of All Models (LLM/Hybrid Severity)

**(Severity Target - LLM/Hybrid)**
*   **Models Shown:** Boundary ds Hybrid, ds simple, ds with examples, ds with examples & refine, ds with examples & refine & base, gpt simple, gpt with examples, gpt with examples & refine, gpt with examples & refine & base.

![Grid of confusion matrices for Severity using LLM methods](placeholder_image_confusion_matrix_severity_2)

---

### Results: Comparison of All Models (Summary)

**Traditional Models:**

*   **Diagnosis:**
    *   Best Model: MorphBoost (stacked GBDT) (All Features)
    *   Accuracy: 0.7821
    *   Precision: 0.7544
    *   Recall: 0.6825
    *   F1-Score: 0.7167

*   **Management:**
    *   Best Model: TreeNet (All Features)
    *   Accuracy: 0.8333
    *   Precision: 0.8013
    *   Recall: 0.8333
    *   F1-Score: 0.8131

*   **Severity:**
    *   Best Model: Neural Decision Forest (All Features)
    *   Accuracy: 0.8846
    *   Precision: 0.9130
    *   Recall: 0.9545
    *   F1-Score: 0.9333

**Best LLM without Model:**

*   **Diagnosis:**
    *   Model: gpt with examples & refine
    *   Accuracy: 0.7115
    *   Precision: 0.6250
    *   Recall: 0.7143
    *   F1-Score: 0.6667

*   **Management:**
    *   Model: ds with examples & refine & base
    *   Accuracy: 0.7500
    *   Precision: 0.8063
    *   Recall: 0.7500
    *   F1-Score: 0.7572

*   **Severity:**
    *   Model: gpt with examples & refine
    *   Accuracy: 0.8141
    *   Precision: 0.9328
    *   Recall: 0.8409
    *   F1-Score: 0.8845

---

### Results: New Feature from deepseek

**Original Feature:**

*   **Diagnosis:**
    *   Best Model: MorphBoost (stacked GBDT) (All Features)
    *   Accuracy: 0.7821
    *   Precision: 0.7544
    *   Recall: 0.6825
    *   F1-Score: 0.7167

*   **Management:**
    *   Best Model: TreeNet (All Features)
    *   Accuracy: 0.8333
    *   Precision: 0.8013
    *   Recall: 0.8333
    *   F1-Score: 0.8131

*   **Severity:**
    *   Best Model: Neural Decision Forest (All Features)
    *   Accuracy: 0.8846
    *   Precision: 0.9130
    *   Recall: 0.9545
    *   F1-Score: 0.9333

**Add New Feature:**

*   **Diagnosis:**
    *   Best Model: DT-GFN (ExtraTrees approx) (All Features)
    *   Accuracy: 0.7821
    *   Precision: 0.7164
    *   Recall: 0.7619
    *   F1-Score: 0.7385

*   **Management:**
    *   Best Model: TEL (Tree Ensemble Layer) (All Features)
    *   Accuracy: 0.8333
    *   Precision: 0.7998
    *   Recall: 0.8333
    *   F1-Score: 0.8148

*   **Severity:**
    *   Best Model: TEL (Tree Ensemble Layer) (All Features)
    *   Accuracy: 0.8846
    *   Precision: 0.9191
    *   Recall: 0.9470
    *   F1-Score: 0.9328

---

## Part 3: Further Analysis on Handling Missing Values

---

### Handling of Missing Value

**List of Missing Features (Partial List):**
*   Appendix_Diameter: 284 missing (36.3%)
*   RBC_in_Urine: 206 missing (26.3%)
*   Ketones_in_Urine: 200 missing (25.6%)
*   WBC_in_Urine: 199 missing (25.4%)
*   Ipsilateral_Rebound_Tenderness: 163 missing (20.8%)
*   Neutrophil_Percentage: 103 missing (13.2%)
*   US_Number: 22 missing (2.8%)
*   Stool: 17 missing (2.2%)
*   Contralateral_Rebound_Tenderness: 15 missing (1.9%)
*   Body_Temperature: 7 missing (0.9%)
*   Age: 1 missing (0.1%)

---

### Handling of Missing Value: Direct method (Median Imputation)

```python
def get_models_with_imputation():
    """
    Return dictionary of models that can handle missing values.
    Uses pipelines with imputation for models that don't natively support NaN.
    """
    # Imputer for models that need complete data
    imputer = SimpleImputer(strategy='median')
```

*   Neglect of the differences between classes.
*   Distortion of Covariance/Correlation.

---

### Handling of Missing Value: KNN Imputation

```python
knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
```

*   Computationally Expensive.
*   Curse of Dimensionality.

---

### Handling of Missing Value: MICE Imputation

**MICE (Multivariate Imputation by Chained Equations)**

*   **Initialization**
*   **Iteration**
    *   $\theta_j^{(t)} \sim P(\theta_j | X_j^{obs}, X_{-j}^{(t)})$
    *   $X_j^{miss(t+1)} \sim P(X_j^{miss} | X_{-j}^{(t)}, \theta_j^{(t)})$
*   **Pooling**
    *   $\bar{\beta} = \frac{1}{m} \sum_{k=1}^m \hat{\beta}_k$

---

### Handling of Missing Value: MICE (Continued)

```python
mice_imputer = IterativeImputer(estimator=BayesianRidge(),
                                max_iter=20,
                                random_state=42)
```

*   Dependency on the "Missing at Random (MAR)".
*   Computational Cost.

---

### Handling of Missing Value: Compare and Discussion

**Median Imputation**

| Target        | Model                | Score              |
| :------------ | :------------------- | :----------------- |
| **Diagnosis** | Logistic Regression  | 0.7692307692307690 |
| **Diagnosis** | Gradient Boosting    | 0.7948717948717950 |
| **Diagnosis** | HistGradientBoosting | 0.7948717948717950 |
| **Severity**  | Logistic Regression  | 0.9358974358974360 |
| **Severity**  | Gradient Boosting    | 0.9358974358974360 |
| **Severity**  | HistGradientBoosting | 0.9423076923076920 |

**KNN Imputation**

| Target        | Model                            | Score              |
| :------------ | :------------------------------- | :----------------- |
| **Diagnosis** | Logistic Regression (GAM-Spline) | 0.7884615384615380 |
| **Diagnosis** | Gradient Boosting (GBDT)         | 0.7820512820512820 |
| **Diagnosis** | Hist Gradient Boosting           | 0.7948717948717950 |
| **Severity**  | Logistic Regression (GAM-Spline) | 0.8717948717948720 |
| **Severity**  | Gradient Boosting (GBDT)         | 0.9294871794871800 |
| **Severity**  | Hist Gradient Boosting           | 0.9358974358974360 |

**MICE Imputation**

| Target        | Model                            | Score              |
| :------------ | :------------------------------- | :----------------- |
| **Diagnosis** | Logistic Regression (GAM-Spline) | 0.7884615384615380 |
| **Diagnosis** | Gradient Boosting (GBDT)         | 0.7948717948717950 |
| **Diagnosis** | Hist Gradient Boosting           | 0.8205128205128210 |
| **Severity**  | Logistic Regression (GAM-Spline) | 0.8846153846153850 |
| **Severity**  | Gradient Boosting (GBDT)         | 0.9230769230769230 |
| **Severity**  | Hist Gradient Boosting           | 0.9294871794871800 |

**Median (use Ultrasound)**

| Target        | Model                | Score              |
| :------------ | :------------------- | :----------------- |
| **Diagnosis** | Logistic Regression  | 0.9230769230769230 |
| **Diagnosis** | Gradient Boosting    | 0.9487179487179490 |
| **Diagnosis** | HistGradientBoosting | 0.9807692307692310 |
| **Severity**  | Logistic Regression  | 0.9166666666666670 |
| **Severity**  | Gradient Boosting    | 0.9615384615384620 |
| **Severity**  | HistGradientBoosting | 0.9423076923076920 |

---

# Thanks for listening!