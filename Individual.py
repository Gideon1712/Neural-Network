
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# extraxt data and normalise it
data_set = pd.read_csv('ce889_dataCollection.csv')
df = pd.DataFrame(data_set)
number_of_epochs = 100
layers = [2,6,2]
early_stop = 10
increased_rmse = 0
successful_epoch = 0

def minmax(data):
    return (data - data.min())/(data.max() - data.min())
normalized_data = minmax(df)

random_data = normalized_data.sample(frac = 1, random_state=42)

train_data, valid_data, test_data = np.split(random_data, [int(.6*len(random_data)), int(.8*len(random_data))])

train_data = train_data.to_numpy()
valid_data =  valid_data.to_numpy()

train_input = train_data[:,:2] 
train_output = train_data[:,-2:] 
valid_input = valid_data[:,:2]
valid_output = valid_data[:,-2]

class NeuralNetwork:
    def __init__(self, training_input, training_output, valid_input, valid_output, lamda, learning_rate, momentum, layers, number_of_epochs):
        self.training_input = training_input
        self.training_output = training_output
        self.number_of_epochs = number_of_epochs
        self.valid_input = valid_input
        self.valid_output = valid_output
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.layers = layers
        self.bias = 0

        self.old_weight_input_hidden = np.zeros((self.layers[0],self.layers[1]))
        self.old_weight_hidden_output = np.zeros((self.layers[1],(self.layers[2])))
        self.old_change_in_weight_input_hidden = np.zeros((self.layers[0],self.layers[1]))
        self.old_change_in_weight_hidden_output = np.zeros((self.layers[1],(self.layers[2])))
        self.updated_weight_input_hidden = np.zeros((self.layers[0],self.layers[1]))
        self.updated_weight_hidden_output = np.zeros((self.layers[1],(self.layers[2])))
        self.updated_hidden_bias = np.zeros(self.layers[2])
        self.updated_input_bias = np.zeros(self.layers[1])
        self.old_change_in_hidden_bias = np.zeros(self.layers[2])
        self.old_change_in_input_bias = np.zeros(self.layers[1])



   
    #calculate the weight multiplication
    def input_func(self, inputs, epoch):
        self.epoch = epoch
        if epoch == 0:
            self.weights_input_hidden = np.random.rand(self.layers[0], self.layers[1])
            self.input_bias = np.random.rand(self.layers[1])
        else:
            self.weights_input_hidden = self.updated_weight_input_hidden
            self.input_bias = self.updated_input_bias

        self.inputs = inputs
        self.v = np.dot(self.inputs, self.weights_input_hidden) + self.input_bias
        return self.v

    #calculate the activation function for the neurons in the hidden layers
    def activation_neurons(self):
        self.hidden_layer = (1/(1 + np.exp(-self.lamda*self.v)))
        return self.hidden_layer

    #calculate the output
    def output_func(self):
        if self.epoch == 0:
            self.weights_hidden_output = np.random.rand(self.layers[1], self.layers[2])
            self.hidden_bias = np.random.rand(self.layers[2])
        else:
            self.weights_hidden_output = self.updated_weight_hidden_output
            self.hidden_bias = self.updated_hidden_bias

        self.y = np.dot(self.hidden_layer, self.weights_hidden_output)
        return self.y

    #calculate the activation function for the neurons in the output layers
    def activation_output(self):
        self.actual_output = (1/(1 + np.exp(-self.lamda*self.y)))
        return self.actual_output

    #calculate the error
    def error_func(self, output):
        self.desired_outputs = output
        self.error = self.desired_outputs - self.actual_output
        self.mean_squared_error = ((np.sum(self.error))**2)/2
        return self.mean_squared_error

#Back Propagation
    #calculate the first gradient
    def calculate_gradient_k(self):
        self.gradients_k = self.error * self.lamda * self.actual_output * (1 - self.actual_output)
        return self.gradients_k

    
    def change_in_hidden_output(self):
        self.change_in_weight_hidden_output = self.learning_rate * np.outer(self.hidden_layer,self.gradients_k) + (self.momentum * self.old_change_in_weight_hidden_output)
        self.old_change_in_weight_hidden_output = self.change_in_weight_hidden_output
        return self.change_in_weight_hidden_output
    

    def update_weight_hidden_output(self):
        self.updated_weight_hidden_output = self.change_in_weight_hidden_output + self.weights_hidden_output
        self.weights_hidden_output = self.updated_weight_hidden_output
        return self.updated_weight_hidden_output
    

    def cal_change_hidden_bias(self):
        self.change_hidden_bias = (self.learning_rate * (1 * self.gradients_k)) + (self.momentum * self.old_change_in_hidden_bias)
        self.old_change_in_hidden_bias = self.change_hidden_bias
        return self.change_hidden_bias

    
    def cal_update_hidden_bias(self):
        self.updated_hidden_bias = self.change_hidden_bias + self.hidden_bias
        self.hidden_bias = self.updated_hidden_bias
        return self.updated_hidden_bias

    
    def calculate_gradient_h(self):
        self.gradients_h = self.lamda * self.hidden_layer * (1 - self.hidden_layer) * np.dot(self.gradients_k, self.weights_hidden_output.T)
        return self.gradients_h

    
    def cal_change_in_weight_input_hidden(self):
        self.change_in_weight_input_hidden = self.learning_rate * np.outer(self.inputs, self.gradients_h) + (self.momentum * self.old_change_in_weight_input_hidden)
        self.old_change_in_weight_input_hidden = self.change_in_weight_input_hidden
        return self.change_in_weight_input_hidden
    
    def update_weight_input_hidden(self):
        self.updated_weight_input_hidden = self.change_in_weight_input_hidden + self.weights_input_hidden
        self.weights_input_hidden = self.updated_weight_input_hidden
        return self.updated_weight_input_hidden
    
    def cal_change_input_bias(self):
        self.change_input_bias = self.learning_rate * (1 * self.gradients_h) + (self.momentum * self.old_change_in_input_bias)
        self.old_change_in_input_bias = self.change_input_bias
        return self.change_input_bias

    
    def cal_updated_input_bias(self):
        self.updated_input_bias = self.change_input_bias + self.input_bias
        self.input_bias = self.updated_input_bias
        return self.updated_input_bias


    #Feed-forward
    def feed_forward(self, inputs, desired_output, epochs):
        self.input_func(inputs, epochs)
        self.activation_neurons()
        self.output_func()
        self.activation_output()
        self.error_func(desired_output)
        return self.activation_output

    #Backpropagation
    def backpropagation(self):
        self.calculate_gradient_k()
        self.change_in_hidden_output()
        self.update_weight_hidden_output()
        self.cal_change_hidden_bias()
        self.cal_update_hidden_bias()
        self.calculate_gradient_h()
        self.cal_change_in_weight_input_hidden()
        self.update_weight_input_hidden()
        self.cal_change_input_bias()
        self.cal_updated_input_bias()


   #for loop for epochs
    def train_nn(self):
        global increased_rmse
        global successful_epoch
        self.train_rmse = []
        self.val_rmse = []
        self.weights = []
        for epochs in range(number_of_epochs):
            train_mse =[]
            val_mse =[]
            for inputs, desired_output in zip(self.training_input, self.training_output):
                self.feed_forward(inputs, desired_output, epochs)
                self.backpropagation()
                train_mse.append(self.mean_squared_error)
            self.train_rmse.append(np.sqrt(np.mean(train_mse)))

            for inputs, desired_output in zip(self.valid_input, self.valid_output):
                self.feed_forward(inputs, desired_output, epochs)
                val_mse.append(self.mean_squared_error)
            self.val_rmse.append(np.sqrt(np.mean(val_mse)))
            #print(self.rmse)
            print(epochs+1)
            
            successful_epoch += 1
            if self.val_rmse[epochs] >= self.val_rmse[epochs - 1]:
                increased_rmse += 1
            if increased_rmse == early_stop:
                break
        self.weights.append(self.update_weight_hidden_output())
        self.weights.append(self.cal_update_hidden_bias())
        self.weights.append(self.update_weight_input_hidden())
        self.weights.append(self.cal_updated_input_bias())
        print(self.weights)
               




train = NeuralNetwork(train_input, train_output, valid_input, valid_output, .8, .1, .9, layers, number_of_epochs)
train.train_nn()

plt.plot(range(successful_epoch),train.train_rmse, label = 'Train Loss')
plt.plot(range(successful_epoch),train.val_rmse, label = 'Validation Loss')
plt.title('Train & Validation RMSE against number of epochs')
plt.ylabel('RMSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()
