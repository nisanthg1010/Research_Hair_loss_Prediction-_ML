# Install if necessary
# pip install imbalanced-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFECV
from imblearn.combine import SMOTETomek

# Load dataset
file_path = "VPN_Hair_Loss.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Target column
target_column = 'Target'
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found.")

X = df.drop(columns=[target_column])
y = df[target_column]

# Handle missing values
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()

if numerical_cols:
    X[numerical_cols] = X[numerical_cols].apply(pd.to_numeric, errors='coerce')
    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].median())

if categorical_cols:
    X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])
    encoder = LabelEncoder()
    for col in categorical_cols:
        X[col] = encoder.fit_transform(X[col])

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection using RFECV
rfe_selector = RFECV(RandomForestClassifier(n_estimators=100, random_state=42), step=1, cv=5, scoring='accuracy')
X_selected = rfe_selector.fit_transform(X_scaled, y)

# Apply SMOTETomek
sm = SMOTETomek(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_selected, y)

# Data augmentation: 4 copies
def augment_data(X, y, noise_level=0.01, copies=4):
    augmented_X = [X]
    augmented_y = [y]
    for _ in range(copies):
        noise = np.random.normal(0, noise_level, X.shape)
        augmented_X.append(X + noise)
        augmented_y.append(y)
    return np.vstack(augmented_X), np.hstack(augmented_y)

X_augmented, y_augmented = augment_data(X_resampled, y_resampled)

# Final train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_augmented, y_augmented, test_size=0.2, stratify=y_augmented, random_state=42
)

# Define models
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)

# Stacking Ensemble Classifier (RF, KNN, GB)
stacking_clf = StackingClassifier(
    estimators=[('rf', rf), ('knn', knn), ('gb', gb)],
    final_estimator=GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    passthrough=True
)

print("\nTraining Ensemble Model 2 (RF, KNN, GB)...")
stacking_clf.fit(X_train, y_train)
y_pred_stack = stacking_clf.predict(X_test)

print("\nEnsemble Model 2 (RF, KNN, GB) Report:")
print(classification_report(y_test, y_pred_stack))
print(f"Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_stack), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Ensemble Model 2 (RF, KNN, GB)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("✅ Ensemble Model 2 evaluation complete.")
