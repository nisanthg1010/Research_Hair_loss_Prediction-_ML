# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Load dataset
file_path = 'VPN_Hair_Loss.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)

# Example synthetic dataset for illustration purposes
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, weights=[0.6, 0.4], random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Apply SMOTE to training and test sets to handle class imbalance
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Model 1: Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Model 2: XGBoost Classifier
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')

# Model 3: LightGBM Classifier
lgbm_model = lgb.LGBMClassifier(random_state=42)

# Hyperparameter Tuning for Random Forest
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}

# Hyperparameter Tuning for XGBoost
xgb_params = {
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Hyperparameter Tuning for LightGBM
lgbm_params = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'num_leaves': [31, 50],
    'subsample': [0.8, 1.0]
}

# GridSearch for Random Forest
grid_rf = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train_resampled, y_train_resampled)

# GridSearch for XGBoost
grid_xgb = GridSearchCV(xgb_model, xgb_params, cv=3, scoring='accuracy', n_jobs=-1)
grid_xgb.fit(X_train_resampled, y_train_resampled)

# GridSearch for LightGBM
grid_lgbm = GridSearchCV(lgbm_model, lgbm_params, cv=3, scoring='accuracy', n_jobs=-1)
grid_lgbm.fit(X_train_resampled, y_train_resampled)

# Stacking Classifier (Combining Random Forest, XGBoost, and LightGBM)
stacking_clf = StackingClassifier(estimators=[
    ('rf', grid_rf.best_estimator_),
    ('xgb', grid_xgb.best_estimator_),
    ('lgbm', grid_lgbm.best_estimator_)
], final_estimator=LogisticRegression())

# Train the Stacking Classifier
stacking_clf.fit(X_train_resampled, y_train_resampled)

# Evaluate the model
y_pred = stacking_clf.predict(X_test)

# Classification Report and Confusion Matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Feature Importance (Optional: You can visualize it if needed)
# Random Forest Feature Importance
rf_model = grid_rf.best_estimator_
feature_importances = rf_model.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importances, color="green", alpha=0.7)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.title("Feature Importances in Random Forest")
plt.show()
