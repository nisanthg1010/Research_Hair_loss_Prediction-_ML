# preprocess_hairloss.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# Step 0: Set correct file path
file_path = 'VPN_Hair_Loss.csv'  # Correct path based on your folder

# Step 1: Load the dataset
if not os.path.exists(file_path):
    print(f"Error: The file '{file_path}' does not exist. Please check the path.")
else:
    try:
        data = pd.read_csv(file_path, encoding='latin1', engine='python', on_bad_lines='skip')  # Encoding and engine fix
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

    # Step 2: Display dataset info
    print("Dataset info:")
    print(data.info())

    # Step 3: Handle missing values
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

    categorical_columns = data.select_dtypes(include=['object']).columns
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

    # Step 4: Encode categorical features
    label_encoders = {}
    for column in categorical_columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Step 5: Scale numerical features
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Step 6: Split the dataset into features (X) and target (y)
    target_column = 'Target'  # Replace if your target is different
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Step 7: Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 8: Show shapes
    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)
    print("Shape of y_train:", y_train.shape)
    print("Shape of y_test:", y_test.shape)

    # Step 9: Save processed files
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    print("\nPreprocessing complete! Files saved.")
