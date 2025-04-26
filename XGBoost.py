# Import necessary libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
file_path = 'VPN_Hair_Loss.csv'  # Replace with actual path
data = pd.read_csv(file_path)

# Using a synthetic dataset for demonstration
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, weights=[0.6, 0.4], random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply ADASYN for better minority class balancing
adasyn = ADASYN(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)

# Scale features using StandardScaler for better performance
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Optimized Hyperparameter Tuning for XGBoost
params = {
    "n_estimators": [300, 500, 700],
    "max_depth": [3, 6, 10],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2],
    "reg_lambda": [1, 2, 5]
}

random_search = RandomizedSearchCV(
    xgb.XGBClassifier(objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss'),
    param_distributions=params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    n_iter=15,
    random_state=42
)

random_search.fit(X_train_resampled, y_train_resampled)

# Train the best model
best_model = random_search.best_estimator_
print(f"Best Parameters: {random_search.best_params_}")

# Evaluate the model
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Plot Feature Importances
feature_importances = best_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importances, color="blue", alpha=0.7)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.title("Feature Importances in XGBoost Classifier")
plt.show()
