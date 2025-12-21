You are a statistician working on a machine learning project regarding child appendicitis. You are given a `.xlsx` file (located in `./data/app_data.xlsx`) which contains tabular data about pediatric patients with suspected appendicitis.

## Data Preprocessing

The first sheet of the excel file contains the detailed patient data. The second sheet contains metadata that describes each column in the first sheet. The target that you are required to predict is the `Diganosis`, `Management` and `Severity` columns. You need to first split the data into train and test sets, then convert the original excel data to `.npy` files, transforming the features to numerical values or one-hot encoding as necessary. You need to faithfully handle any missing values. Then, output two versions of the processed data: one with all features and the other dropping features belonging to the `Ultrasound` category as described in the metadata sheet. The data should be saved in the `./processed_data/` directory.

Output a summary of the transformations you applied to the columns, including how missing values were handled and the encoding methods used in a separate text file inside the `./processed_data/` directory.

## Preliminary Analysis

Perform a preliminary analysis of the processed data. Plot the correlations trends between features and the target variables. You can use visualizations such as heatmaps, scatter plots, or bar charts to illustrate these relationships. Save these plots in the `./analysis/` directory.

## Fitting Models

Fit at least three different statistical or machine learning models to predict the target variables (`Diganosis`, `Management`, and `Severity`). You need to fit models for both with and without ultrasound features. You can choose from models such as GAE, Random Forest, Gradient Boosting, or any other suitable algorithms. You will need to carefully handle the missing values. You may explore different ways of handling missing values. Evaluate the performance of each model using appropriate metrics (e.g., accuracy, precision, recall, F1-score, GCV) and compare their results. After that, perform feature selection to further enhance your models. You may save your results to the `./models/` directory.

## Requirements

An conda environment is already available as `stat-learning`, which is based on `python 3.7`. You need to first run `conda activate stat-learning` to activate the environment before executing any scripts. You may install any additional packages if necessary using `pip` or `conda install`.

## Report

Compile a comprehensive report summarizing your data preprocessing steps, preliminary analysis findings, model fitting procedures, and evaluation results. Include visualizations and tables where appropriate to support your analysis. Save this report as a `.md` file in the root directory of the project. You should also include a takeaway section highlighting the key results you obtained and the data that you preprocessed to be used for future modeling tasks.