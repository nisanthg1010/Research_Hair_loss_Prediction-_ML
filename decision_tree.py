# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from rich import print as rprint # Import rich's print function
from rich.text import Text # Import Text from rich

# Load your dataset
rprint(Text("Loading the dataset...", style="bold blue")) # Use rich's print
file_path = 'VPN_Hair_Loss.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)
rprint(Text("Dataset loaded successfully!", style="bold green"))# Use rich's print
# Replace this with your actual dataset loading step
# For demonstration, I'm using a synthetic dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.7, 0.3], random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 1: Augment the Dataset (Handle Class Imbalance with SMOTE)
print("Before SMOTE:")
print(f"Class Distribution: {np.bincount(y_train)}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("After SMOTE:")
print(f"Class Distribution: {np.bincount(y_train_resampled)}")

# Step 2: Standardize the Features
scaler = StandardScaler()
X_train_resampled = scaler.fit_transform(X_train_resampled)
X_test = scaler.transform(X_test)

# Step 3: Tune Hyperparameters for Decision Tree using Grid Search
params = {
    "max_depth": [3, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=params,
    cv=5,
    scoring="accuracy"
)
grid_search.fit(X_train_resampled, y_train_resampled)

# Step 4: Train the Best Model
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Step 5: Evaluate the Model
y_pred = best_model.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Step 6: Visualize the Results (if needed)
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(best_model, filled=True, feature_names=[f"Feature_{i}" for i in range(X.shape[1])])
plt.title("Decision Tree Visualization")
plt.show()