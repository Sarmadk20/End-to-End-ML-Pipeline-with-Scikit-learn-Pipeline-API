# End-to-End-ML-Pipeline-with-Scikit-learn-Pipeline-API

# Objective of the Task

The objective of this task is to design and implement a reusable, scalable, and production-ready machine learning pipeline to predict customer churn using the Telco Customer Churn dataset.
The pipeline automates data preprocessing, model training, hyperparameter tuning, and model serialization using Scikit-learn’s Pipeline API.

# Methodology / Approach
1. Data Preparation

Used the Telco Churn dataset containing customer demographics, services, charges, and behavioral attributes.

Removed non-informative and leakage-prone columns such as:

Customer ID

Customer Status

Churn Category

Churn Reason

Converted the target variable Churn Label into binary form:

Yes → 1, No → 0

2. Data Preprocessing (Pipeline-Based)

Numerical Features

Missing values handled using median imputation

Standardized using StandardScaler

Categorical Features

Missing values handled using most frequent imputation

Encoded using OneHotEncoder

Used ColumnTransformer to apply transformations in parallel and avoid data leakage.

3. Model Training

Two machine learning models were implemented within pipelines:

Logistic Regression

Acts as a baseline and provides interpretability

Random Forest Classifier

Captures non-linear patterns and feature interactions

4. Hyperparameter Tuning

Used GridSearchCV with 5-fold cross-validation

Optimized model performance based on accuracy

Selected the best-performing estimator automatically

5. Model Export

The complete pipeline (preprocessing + trained model) was exported using joblib

Enables seamless deployment and inference without manual preprocessing

# Key Results / Observations

Pipeline-based preprocessing prevents data leakage and ensures consistency.

Random Forest generally outperformed Logistic Regression due to its ability to model complex relationships.

The final saved pipeline can directly accept raw customer data and produce churn predictions.

The approach is fully reusable, scalable, and production-ready.

Accuracy: 0.9659332860184529

# Technologies Used

Python

Scikit-learn

Pandas

Joblib

# Output

Trained model pipeline saved as:
telco_churn_pipeline.pkl