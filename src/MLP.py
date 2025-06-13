import numpy as np
from src.utils import sigmoid, sigmoid_derivative, tanh, tanh_derivative

class MLP:
    def __init__(self, inputSize=5,hiddenLayers=2, neurons=3,outputSize=3, learning_rate=0.01, epochs=1000, Abias=True, activationFn=None):
        self.inputSize = inputSize
        self.hiddenLayers = hiddenLayers
        self.neurons = neurons
        self.outputSize = outputSize
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.Abias = Abias
        self.activationFn = sigmoid if activationFn == "sigmoid" else tanh
        self.actvFnDerivative = sigmoid_derivative if activationFn == "sigmoid" else tanh_derivative
        
        self.bias = [] if self.Abias else None 
        self.weights = []  
        
        self.weights.append(np.random.uniform(0, 0.5, size=(inputSize, neurons[0])))  # input weights initialization
        
        if self.Abias:
            self.bias.append(np.zeros((1, neurons[0])))

        for i in range(1, hiddenLayers):                                # hidden layers weights initialization
                    self.weights.append(np.random.randn(neurons[i-1], neurons[i]) * 0.01)
                    if self.Abias:
                        self.bias.append(np.zeros((1, neurons[i])))
                        
        self.weights.append(np.random.randn(neurons[-1], 3) * 0.01)        # output weights initialization
        if self.Abias:
            self.bias.append(np.zeros((1, 3)))
            
    # forward propagation function code
    def forwardPass(self, X):
        layerOutput = [X]

        for i in range(len(self.weights)):
            X = np.dot(X, self.weights[i])
            if self.Abias:
                X += self.bias[i]
            X = self.activationFn(X)
            layerOutput.append(X)

        return layerOutput
    
    # backward propagation function code
    def backwardPass(self,x,y,outputs):
        error = [outputs[-1] - y]
        for i in range(len(self.weights) - 1, 0, -1):
            errorSignal = np.dot(error[-1], self.weights[i].T) * self.actvFnDerivative(outputs[i])
            error.append(errorSignal)

        error.reverse()  # reversing the layers order

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(outputs[i].T, error[i])
            if self.Abias:
                self.bias[i] -= self.learning_rate * np.sum(error[i], axis=0, keepdims=True)

        
    def train(self, X_train, y_train):
        y_trainEncoded = np.eye(self.outputSize)[y_train]

        for epoch in range(self.epochs):
            outputs = self.forwardPass(X_train)
            self.backwardPass(X_train, y_trainEncoded, outputs)

            if epoch % 50 == 0:
                errorsSquared = np.square(y_trainEncoded - outputs[-1])
                loss = np.mean(np.sum(errorsSquared, axis=1)) 
                
                predicted_labels = np.argmax(outputs[-1], axis=1)
                true_labels = np.argmax(y_trainEncoded, axis=1)
                accuracy = np.mean(predicted_labels == true_labels)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")

        
        
    def test(self, X_test, y_test):
        y_testEncoded = np.eye(self.outputSize)[y_test]

        for epoch in range(self.epochs):
            outputs = self.forwardPass(X_test)
            self.backwardPass(X_test, y_testEncoded, outputs)

            if epoch % 50 == 0:
                errorsSquared = np.square(y_testEncoded - outputs[-1])
                loss = np.mean(np.sum(errorsSquared, axis=1))
                
                predicted_labels = np.argmax(outputs[-1], axis=1)
                true_labels = np.argmax(y_testEncoded, axis=1)
                accuracy = np.mean(predicted_labels == true_labels)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy*100:.2f}%")    
                            
    # sample classification
    def predict(self, sample):
        return self.forwardPass(sample)[-1]
