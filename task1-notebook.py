import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
import random

data=pd.read_csv("birds.csv")

sns.histplot(data['fin_length'],bins=25)

gender_counts = data['gender'].value_counts()

# Plot pie chart
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['blue', 'red'], startangle=140)
plt.title('Gender Distribution')
plt.show()

sns.histplot(data['body_mass'],kde=True) # right skewed
plt.show()

sns.histplot(data['beak_depth'],kde=True) # left skewed
plt.show()

sns.histplot(data['fin_length'],kde=True) 
plt.show()

Category_counts = data['bird category'].value_counts()
plt.pie(Category_counts, labels=Category_counts.index, autopct='%1.1f%%', colors=['blue', 'red','yellow'], startangle=140)
plt.title('Gender Distribution')
plt.show()

data.shape # (150, 6)
data.isnull().sum() #6 nulls in gender , we have to encode the gender and category
gender_mappig={'male':0,'female':1}
data["gender"]=data["gender"].replace(gender_mappig)
categories_mapping={'A':0,'B':1,'C':2}
data['bird category'] = data['bird category'].replace(categories_mapping)
(data["gender"]==0).sum()#females=77,male=73
data["gender"] = data["gender"].bfill() #backkfilling the nulls
data.isnull().sum() 

plt.figure(figsize=(12, 8))  # Set figure size
for idx, i in enumerate(data.columns, start=1):
    plt.subplot(2, 3, idx)  # 2 rows, 3 columns
    sns.boxplot(data=data, x=i)
    plt.title(f'Plot of {i}')
    plt.xlabel(i)
    plt.ylabel("Values")

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

scaler=StandardScaler() # to preserve the data shape
data_scaled=data.columns[1:-1]
data[data_scaled]=scaler.fit_transform(data[data_scaled])

from sklearn.model_selection import train_test_split
x=data.drop(columns=['bird category']).values
y=data['bird category'].values


feature1 = "gender"
feature2 = "beak_length"  
new_df = data[[feature1, feature2]].copy()
new_df["bird category"] = data["bird category"]

X=new_df.drop(columns='bird category')
Y=new_df['bird category']

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
    
class_1 = 0  
class_2 = 1  
filtered_data = filter_classes(data, class_1, class_2)
print(filtered_data['bird category'].value_counts())

def train_split(selected_features,filtered_data):
    # Select a balanced subset of 30 samples from each bird category class
    train_data = filtered_data.groupby('bird category', group_keys=False).apply(lambda x: x.sample(n=30, random_state=42)).reset_index(drop=True)    
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
selected_features = ['gender', 'beak_length'] 
# train_split(selected_features,filtered_data) 
X_train,y_train,X_test,y_test=train_split(selected_features,filtered_data) 


x=round(random.uniform(0,0.5),2)
#function returns a random floating-point number N such that a<=N<b
#it generates a number between 0 and 0.5.

class SLP:
    def __init__(self, learning_rate=0.01, epochs=100, Abias=True):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.Abias = Abias
        self.bias = 0 if self.Abias else None # else 1 ?
        self.weights = None  

    def train(self, X_train, y_train):
        number_of_samples, number_of_features = X_train.shape
        self.weights = np.random.uniform(0, 0.5, size=number_of_features) 
        #If number_of_features = 3, then np.random.uniform(0, 0.5, size=3) might generate something like [0.12, 0.34, 0.25]

        for epoch in range(self.epochs):
            correct = 0
            for i in range(number_of_samples):
                Net_input = np.dot(X_train.iloc[i, :], self.weights) + (self.bias if self.Abias else 0)
                Y_prediction = 1 if Net_input > 0 else -1 #signum
                error = y_train.iloc[i] - Y_prediction 
                self.weights += self.learning_rate * error * X_train.iloc[i, :] #update weights
                if self.Abias:
                    self.bias += self.learning_rate * error

                if error == 0:
                    correct += 1
            
            accuracy = correct / number_of_samples * 100
            # print(f"Epoch {epoch+1}/{self.epochs} - SLP Training Accuracy: {accuracy:.2f}%")
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}- SLP Training Accuracy: {accuracy:.2f}%")

    def test(self, X_test, y_test):
        number_of_samples = X_test.shape[0]
        correct = 0
        for i in range(number_of_samples):
            Net_input = np.dot(X_test.iloc[i, :], self.weights) + (self.bias if self.Abias else 0)
            Y_prediction = 1 if Net_input > 0 else -1
            if Y_prediction == y_test.iloc[i]:
                correct += 1

        accuracy = correct / number_of_samples * 100
        print(f"SLP Test Accuracy: {accuracy:.2f}%")
    def predict(self, X):
       X = np.array(X)
       Net_input = np.dot(X, self.weights) + (self.bias if self.Abias else 0)
       predicted_class = np.where(Net_input > 0, 1, -1)
       class_original= class_1 if predicted_class==-1 else class_2
       return class_original

model = SLP(learning_rate=0.000005, epochs=800, Abias=True)
model.train(X_train, y_train)
model.test(X_test, y_test)
test=np.array([0,-1.1928016115293976])

#different from the SLP as here there is Cost Function that is calc there as simple error
#implement gradient descent
class Adaline:
  def __init__(self, learning_rate=0.01, epochs=100, Abias=True, MSE_thres=0.2):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.Abias = Abias
    self.bias = 0 if self.Abias else None # else 1 ?
    self.weights = None  
    self.losses = []  #loss after avg
    self.MSE_thres=MSE_thres

  def train(self, X_train, y_train):
    number_of_samples, number_of_features = X_train.shape
    # self.weights = np.random.randn(number_of_features) * 0.01
    self.weights = np.random.uniform(0, 0.5, size=number_of_features)

    #If number_of_features = 3, then np.random.uniform(0, 0.5, size=3) might generate something like [0.12, 0.34, 0.25]

    for epoch in range(self.epochs):
      correct = 0
      # losses_epoch = []
      losses_epoch=0
      for i in range(number_of_samples):
        linear_output = np.dot(X_train.iloc[i, :], self.weights) + (self.bias if self.Abias else 0)
        Y_prediction = 1 if linear_output > 0 else -1 
        error = y_train.iloc[i] - linear_output # loss
        #update the weights and bias
        self.weights += self.learning_rate * error * X_train.iloc[i, :] 
        if self.Abias:
          self.bias += self.learning_rate * error
        losses_epoch+=(error**2) #cost function
         # Check if prediction is correct
        if Y_prediction == y_train.iloc[i]:
            correct += 1
      # losses_epoch(list): kol sample
      # mean_loss(variable): avg el losses_epoch -> for one epoch
      # losses(list): contains all the mean_loss of epochs
      mean_loss=(losses_epoch)/number_of_samples
      self.losses.append(mean_loss)
      accuracy = correct / number_of_samples * 100
      # print(f"Epoch {epoch+1}/{self.epochs},  Loss: {mean_loss:.4f} - Training Accuracy: {accuracy:.2f}%")
      if epoch % 10 == 0 or mean_loss <= self.MSE_thres:
        print(f"Epoch {epoch+1}/{self.epochs}, Loss: {mean_loss:.4f} - Adaline Training Accuracy: {accuracy:.2f}%")

      #stopping condition of the MSE
      if mean_loss<= self.MSE_thres:
        break
  
  def test(self, X_test, y_test):
    number_of_samples = X_test.shape[0]
    correct = 0
    for i in range(number_of_samples):
      linear_output = np.dot(X_test.iloc[i, :], self.weights) + (self.bias if self.Abias else 0)
       # Check if prediction is correct
      Y_prediction = 1 if linear_output > 0 else -1
      if Y_prediction == y_test.iloc[i]:
        correct += 1


    accuracy = correct / number_of_samples * 100
    print(f"Adaline Test Accuracy: {accuracy:.2f}%")

  def predict(self, X):
    X = np.array(X)
    Net_input = np.dot(X, self.weights) + (self.bias if self.Abias else 0)
    predicted_class = np.where(Net_input > 0, 1, -1)
    # class_original= class_1 if predicted_class==-1 else class_2
    return predicted_class

model = Adaline(learning_rate=0.000005, epochs=800, Abias=True,MSE_thres=0.2)
model.train(X_train, y_train)
model.test(X_test, y_test)

def plot_decision_boundary(model, X, y):
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
    plt.title("Adaline Decision Boundary")
    plt.legend()
    plt.show()


plot_decision_boundary(model, X_train, y_train)