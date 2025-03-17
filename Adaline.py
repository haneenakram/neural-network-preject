import numpy as np

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
    confusion_matrix = np.zeros((2, 2)) 
    for i in range(number_of_samples):
      linear_output = np.dot(X_test.iloc[i, :], self.weights) + (self.bias if self.Abias else 0)
      Y_prediction = 1 if linear_output > 0 else -1
      if Y_prediction == y_test.iloc[i]:
        correct += 1
        # Actual label
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

    accuracy = correct / number_of_samples * 100
    return confusion_matrix,accuracy

  def predict(self, X):
    X = np.array(X)
    Net_input = np.dot(X, self.weights) + (self.bias if self.Abias else 0)
    predicted_class = np.where(Net_input > 0, 1, -1)
    return predicted_class