import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from sklearn.model_selection import train_test_split

scaler = StandardScaler()

def load_data(filename):
  return pd.read_csv(filename)

print(load_data('birds.csv'))

def scale_data(data, fit=True):
  """Scales the dataset using StandardScaler.

  Parameters:
  - data: DataFrame containing features to scale.
  - fit: If True, fits the scaler on the data (used during training); 
        if False, only transforms (used during prediction).

  Returns:
  - Scaled DataFrame
  """
  global scaler
  feature_columns = data.columns[1:-1]  # Select columns to scale (excluding target)

  if fit:
    data[feature_columns] = scaler.fit_transform(data[feature_columns])
  else:
    data[feature_columns] = scaler.transform(data[feature_columns])

  return data

def preprocessing(data):
  gender_mappig={'male':0,'female':1}
  data["gender"]=data["gender"].replace(gender_mappig)
  categories_mapping={'A':0,'B':1,'C':2}
  data['bird category'] = data['bird category'].replace(categories_mapping)
  data["gender"] = data["gender"].bfill() #backkfilling the nulls

  # Apply scaling
  data = scale_data(data, fit=True)


  x=data.drop(columns=['bird category']).values
  y=data['bird category'].values
  return(data)

def choosing_features(data, feature1 = "gender",feature2 = "beak_length"  ):
  new_df = data[[feature1, feature2]].copy()
  new_df["bird category"] = data["bird category"]

  X=new_df.drop(columns='bird category')
  Y=new_df['bird category']
  return new_df

def filter_classes(data, class_1, class_2):
  """Filters the dataset to include only two selected classes."""
  filtered_data = data[data['bird category'].isin([class_1, class_2])].reset_index(drop=True) 
  # selects only rows where the 'bird category' column contains either class_1 or class_2.
  # The isin() function checks if each value in the 'bird category' column is in the list [class_1, class_2].
  # reset_index(drop=True) resets the row indices of the filtered dataset, ensuring they are sequential and removing the old indices.
  filtered_data['bird category'] = np.where(
      filtered_data['bird category'] == class_1, -1, 1
  )

  return filtered_data
    
# class_1 = 0  
# class_2 = 1  
# filtered_data = filter_classes(data, class_1, class_2)

def train_split(selected_features,filtered_data):
  # Select a balanced subset of 30 samples from each bird category class
  train_data = filtered_data.groupby('bird category', group_keys=False).apply(lambda x: x.sample(n=30, random_state=42))   
  if 'bird category' not in selected_features:
      selected_features.append('bird category')
  train_data = train_data[selected_features]
  test_data = filtered_data[selected_features].drop(train_data.index)
  X_train = train_data.drop(columns=['bird category']) 
  y_train = train_data['bird category']  
  X_test = test_data.drop(columns=['bird category'])
  y_test = test_data['bird category']  
  print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
  print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
  return X_train,y_train,X_test,y_test

# selected_features = ['gender', 'beak_length'] 
# X_train,y_train,X_test,y_test=train_split(selected_features,filtered_data) 

def plot_decision_boundary(model, X, y,feature1,feature2):
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
    
def preprocess_and_scale_input(input_data, feature_1, feature_2):
    """Preprocesses input data by scaling only non-gender features."""
    
    input_data = np.array(input_data, dtype=float).reshape(1, -1)  # Ensure correct shape

    # Check which features need to be scaled
    if feature_1 == 'gender' and feature_2 == 'gender':
        return input_data  # No scaling needed, return as is
    
    elif feature_1 == 'gender':
        # Only scale feature 2
        input_data[:, 1:] = scaler.transform(input_data[:, 1:].reshape(-1, 1))
    
    elif feature_2 == 'gender':
        # Only scale feature 1
        input_data[:, :1] = scaler.transform(input_data[:, :1].reshape(-1, 1))
    
    else:
        # Scale both features
        input_data = scaler.transform(input_data)

    return input_data

def predict(model, input_data,feature_1,feature_2):
  """Predicts the bird category based on user input."""
  processed_input = preprocess_and_scale_input(input_data,feature_1,feature_2)
  prediction = model.predict(processed_input)
  return prediction[0]
