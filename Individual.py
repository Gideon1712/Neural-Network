import numpy as np
from numpy import random
import pandas as pd

#calculate minmaxs, and return numpy array
def minmaxs(df):
    min = df.min
    max = df.max
    scaled_df = (df - min)/(max-min)
    return scaled_df.to_numpy()

class NeuralNetwork:
    def __init__(self, inputs, number_h_n, desired_outputs, learning_rate, momentum):
        self.inputs = inputs
        self.number_h_n = number_h_n
        self.desired_outputs = desired_outputs
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.inputs = np.array(self.inputs)
        self.weights = random.rand(self.number_h_n, len(self.inputs))

    #Feed-forward
    #calculate the weight multiplication
    def weight_multi(self):
        self.v = np.matmul(self.weights, self.inputs)
        return self.v

    #calculate the activation function for the hidden layers
    def activation_neurons(self):
        self.hidden_layer = (1/(1 + np.exp(-self.learning_rate*self.v)))
        return self.hidden_layer

    #calculate the output
    def output_func(self):
        self.y = np.matmul(self.weights, self.inputs)
        return self.y

    #calculate the activation function for the outputs
    def activation_output(self):
        self.output = (1/(1 + np.exp(-self.learning_rate*self.y)))
        return self.output

    #calculate the error
    def error_func(self):
        self.err = self.desired_outputs - self.output
        return self.err

#Back Propagation

    #calculate the first gradient
    def calculate_gradient_k(self):
        self.gradients_k = self.error_func() * self.learning_rate * self.activation_output() * (1 - self.activation_output())
        return self.gradients_k

    #calculate the first change_in_weight
    def cal_change_in_weight_k(self):
        self.change_in_weight_k = self.learning_rate * self.calculate_gradient_k() * self.hidden_layer + (self.momentum * self.old_change_in_weight)
        return self.change_in_weight_k

    #calculate the first updated weights
    def cal_updated_weight_k(self):
        self.update_weight_k = self.change_in_weight_k() + self.old_change_in_weight
        return self.updated_weight_k

    #calculate the new gradient
    def calculate_gradient_h(self):
        self.gradients_h = self.learning_rate * self.activation_output() * (1 - self.activation_output()) * (self.gradients_k())
        return self.gradients_h

    #calculate the second change in weight
    def cal_change_in_weight_h(self):
        self.change_in_weight_h = self.learning_rate * self.calculate_gradient_h() * self.inputs + (self.momentum * self.old_change_in_weight)
        return self.change_in_weight_h

    #calculate the second updated weights
    def cal_updated_weight_h(self):
        self.update_weight_h = self.change_in_weight_h() + self.old_change_in_weight
        return self.updated_weight_h



n1 = NeuralNetwork([1, 2, 3], 2, [1, 3], 0.7, 0.6)
n1.weight_multi()
n1.activation_neurons()
n1.output_func()
n1.activation_output()
print(n1.error_func())


