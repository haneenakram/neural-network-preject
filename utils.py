import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def read_file(file_path):
    data=pd.read_csv(file_path)

def data_encode(data):
    gender_mapping = {'male': 0, 'female': 1}
    data["gender"] = data["gender"].replace(gender_mapping).infer_objects(copy=False)
    categories_mapping = {'A': 0, 'B': 1, 'C': 2}
    data['bird category'] = data['bird category'].replace(categories_mapping).infer_objects(copy=False)
    data["gender"] = data["gender"].bfill() #backkfilling the nulls

def standardize_data(data):
    scaler = StandardScaler()
    data_scaled=data.columns[1:-1]
    data[data_scaled]=scaler.fit_transform(data[data_scaled])    
    return data

def filter_classes(data, class_1, class_2,class_3):
    filtered_data = data[data['bird category'].isin([class_1, class_2,class_3])].reset_index(drop=True)
    class_mapping = {class_1: 0, class_2: 1, class_3: 2}
    filtered_data['bird category'] = filtered_data['bird category'].map(class_mapping)
    
    return filtered_data

def train_test_split_data(filtered_data):
    # X = data.iloc[:, 1:-1]
    # y = data.iloc[:, -1]
    X=filtered_data.drop(columns=['bird category']).values
    Y=filtered_data['bird category'].values 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
    return X_train, X_test, y_train, y_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# decision boundary fn
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
def plot_classification_report(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, fmt=".2f", cmap="Blues")
    plt.title('Classification Report')
    plt.show()
def plot_accuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=['Accuracy'], y=[accuracy], palette='Blues')
    plt.title('Model Accuracy')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.show()
#########################################################################################################################3
