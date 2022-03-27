import numpy as np
import matplotlib.pyplot as plt

##########################
### DATASET
##########################




weights = np.array([[1],[2],[3],[4]], dtype=np.float)
zeros = np.zeros((4, 1), dtype=np.float)
bias = np.array([1] , dtype=np.float)
zero = np.zeros(1, dtype=np.float)

print(weights)
print(zeros)
print(bias)
print(zero)

class Perceptron():
    def __init__(self, num_features, learning_rate, weights, bias):
        self.num_features = num_features
        # self.weights = np.zeros((num_features, 1), dtype=np.float)
        # self.bias = np.zeros(1, dtype=np.float)
        self.weights = weights
        self.bias = bias
        self.learning_rate = learning_rate

    def forward(self, x):
        linear = np.dot(x, self.weights) + self.bias 
        predictions = np.where(linear > 0., 1, 0)
        return predictions
        
    def backward(self, x, y):  
        predictions = self.forward(x)
        errors = self.learning_rate*(y - predictions)
        return errors
        
    def train(self, x, y, epochs):
        for e in range(epochs):
            
            for i in range(y.shape[0]):
                errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
                self.weights =self.weights+ (errors * x[i]).reshape(self.num_features, 1)
                self.bias =self.bias+ errors
                
    def evaluate(self, x, y):
        predictions = self.forward(x).reshape(-1)
        accuracy = np.sum(predictions == y) / y.shape[0]
        return accuracy



