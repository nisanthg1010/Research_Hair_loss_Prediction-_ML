# Import necessary libraries
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import ADASYN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
file_path = 'VPN_Hair_Loss.csv'  # Replace with actual path
data = pd.read_csv(file_path)

# Using a synthetic dataset for demonstration
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, weights=[0.6, 0.4], random_state=42)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Check class distribution before applying ADASYN
print("Original class distribution:", Counter(y_train))

# Apply ADASYN for better minority class balancing
try:
    adasyn = ADASYN(sampling_strategy='auto', random_state=42)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
    print("New class distribution:", Counter(y_train_resampled))
except ValueError as e:
    print(f"ADASYN skipped due to: {e}")
    X_train_resampled, y_train_resampled = X_train, y_train  # Use original data if ADASYN fails

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# **Define Individual Models**
gbm = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# **Train GBM & RF models separately for feature importance extraction**
gbm.fit(X_train_resampled, y_train_resampled)
rf.fit(X_train_resampled, y_train_resampled)

# **Create Voting Classifier**
ensemble_model = VotingClassifier(
    estimators=[('GBM', gbm), ('RF', rf), ('KNN', knn)],
    voting='soft'  # Soft voting improves performance
)

# Train the Ensemble Model
ensemble_model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred_ensemble = ensemble_model.predict(X_test)

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ensemble))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_ensemble))

accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
print(f"\nEnsemble Accuracy: {accuracy_ensemble * 100:.2f}%")

# **Plot Feature Importance (Fixed)**
plt.figure(figsize=(10, 6))
feature_importances = (gbm.feature_importances_ + rf.feature_importances_) / 2  # Averaged from trained models
plt.bar(range(X.shape[1]), feature_importances, color="purple", alpha=0.7)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.title("Feature Importance in Ensemble Model")
plt.show()
