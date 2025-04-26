# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

# Load dataset
file_path = 'VPN_Hair_Loss.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)

# Using a synthetic dataset for demonstration
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, weights=[0.6, 0.4], random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE to training and test sets
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale features using RobustScaler
scaler = RobustScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Optimized Hyperparameter Tuning
params = {
    "n_estimators": [100, 150],
    "max_depth": [None, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "criterion": ["gini", "entropy"],
    "bootstrap": [True, False]
}

random_search = RandomizedSearchCV(
    ExtraTreesClassifier(random_state=42, n_jobs=-1),
    param_distributions=params,
    cv=3,  # Reduced from 5 to 3
    scoring="accuracy",
    n_jobs=-1,
    n_iter=10,  # Randomly tests 10 combinations instead of all
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
plt.bar(range(X.shape[1]), feature_importances, color="green", alpha=0.7)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.title("Feature Importances in Extra Trees Classifier")
plt.show()
