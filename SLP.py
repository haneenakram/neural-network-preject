import numpy as np

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
        confusion_matrix = np.zeros((2, 2)) 
        for i in range(number_of_samples):
            Net_input = np.dot(X_test.iloc[i, :], self.weights) + (self.bias if self.Abias else 0)
            Y_prediction = 1 if Net_input > 0 else -1
            if Y_prediction == y_test.iloc[i]:
                correct += 1
        actual_label = y_test.iloc[i]
        # Update confusion matrix
        if Y_prediction == 1 and actual_label == 1:
            confusion_matrix[0, 0] += 1  # True Positive (TP)
        elif Y_prediction == -1 and actual_label == 1:
            confusion_matrix[0, 1] += 1  # False Negative (FN)
        elif Y_prediction == 1 and actual_label == -1:
            confusion_matrix[1, 0] += 1  # False Positive (FP)
        else:
            confusion_matrix[1, 1] += 1  # True Negative (TN)

        print(f"SLP Test Accuracy: {accuracy:.2f}%")
        accuracy = correct / number_of_samples * 100
        return confusion_matrix,accuracy

    def predict(self, X):
        X = np.array(X)
        Net_input = np.dot(X, self.weights) + (self.bias if self.Abias else 0)
        predicted_class = np.where(Net_input > 0, 1, -1)
        return predicted_class