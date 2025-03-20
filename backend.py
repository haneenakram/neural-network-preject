import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filename):
    return pd.read_csv(filename)

def preprocessing(data):
    gender_mapping = {'male': 0, 'female': 1}
    data["gender"] = data["gender"].replace(gender_mapping)
    categories_mapping = {'A': 0, 'B': 1, 'C': 2}
    data['bird category'] = data['bird category'].replace(categories_mapping)
    data["gender"] = data["gender"].bfill()  
    
    return data

def choosing_features(data, feature1="gender", feature2="beak_length"):
    new_df = data[[feature1, feature2]].copy()
    new_df["bird category"] = data["bird category"]
    return new_df

def filter_classes(data, class_1, class_2):
    filtered_data = data[data['bird category'].isin([class_1, class_2])].reset_index(drop=True)
    filtered_data['bird category'] = np.where(
        filtered_data['bird category'] == class_1, -1, 1
    )
    return filtered_data

def train_split(selected_features, filtered_data):
    # Select balanced subset
    train_data = filtered_data.groupby('bird category', group_keys=False).apply(
        lambda x: x.sample(n=30, random_state=42)
    )
    
    if 'bird category' not in selected_features:
        selected_features.append('bird category')
    
    # Split data
    test_data = filtered_data[selected_features].drop(train_data.index)
    X_train = train_data.drop(columns=['bird category'])
    y_train = train_data['bird category']
    X_test = test_data.drop(columns=['bird category'])
    y_test = test_data['bird category']
    
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test

def plot_decision_boundary(model, X, y, feature1, feature2):
    # Extract feature limits
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    # Generate a range of x values
    x_values = np.linspace(x_min, x_max, 100)

    # Extract weights and bias
    W_i, W_j = model.weights[0], model.weights[1]
    b = model.bias if hasattr(model, 'bias') else 0

    # Compute decision boundary (X_j)
    y_values = -(W_i * x_values + b) / W_j  # From equation: W_i * X_i + W_j * X_j + b = 0

    # Plot decision boundary
    plt.plot(x_values, y_values, 'k--', linewidth=2, label="Decision Boundary")

    # Plot data points with different colors
    plt.scatter(X.iloc[:, 0][y == -1], X.iloc[:, 1][y == -1], color='blue', marker='o', label='Class -1')
    plt.scatter(X.iloc[:, 0][y == 1], X.iloc[:, 1][y == 1], color='red', marker='s', label='Class 1')

    # Labels and title
    plt.xlabel(f"{feature1}")
    plt.ylabel(f"{feature2}")
    plt.title(" Decision Boundary")
    plt.legend()
    plt.show()