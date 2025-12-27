import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('F:\\AI&ML\\Task 2\\telco.csv')
print(df.head())

# Drop identifier and leakage columns
drop_cols = [
    "Customer ID",
    "Customer Status",
    "Churn Category",
    "Churn Reason"
]

X = df.drop(columns=drop_cols + ["Churn Label"])
y = df["Churn Label"].map({"Yes": 1, "No": 0})

# Feature Separation
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns


# Preprocessing Pipelines
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, num_features),
    ("cat", categorical_pipeline, cat_features)
])

# Train-Test Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

# Model Pipelines
lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(max_iter=1000))
])

lr_params = {
    "model__C": [0.01, 0.1, 1, 10]
}

lr_grid = GridSearchCV(
    lr_pipeline,
    lr_params,
    cv=5,
    scoring="accuracy"
)

lr_grid.fit(X_train, y_train)

# Random Forest Classifier Pipeline
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=101))
])

rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

rf_grid = GridSearchCV(
    rf_pipeline,
    rf_params,
    cv=2,
    scoring="accuracy",
    n_jobs=-1
)

rf_grid.fit(X_train, y_train)

# Evaluate Models
best_model = rf_grid.best_estimator_

y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Save the best model
joblib.dump(best_model, "telco_churn_pipeline.pkl")

# Evaluate Models
model = joblib.load("telco_churn_pipeline.pkl")

sample = X_test.iloc[:1]
prediction = model.predict(sample)

print("Churn Prediction:", "Yes" if prediction[0] == 1 else "No")



