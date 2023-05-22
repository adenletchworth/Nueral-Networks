import numpy as np

def simpleNetwork():
    # we have these inputs for our model
    inputs = [1,3,4]

    # we have these three nuerons each with 3 weights
    neurons = [[.5,.1,.7],[.3,.4,-.2],[0,.1,.3]]
    # we have the associated bias for each neuron
    biases = [3,1,2]
    # initalize outputs arrays
    outputs = []
    # loop through all of our weights
    for weights, bias in zip(neurons,biases):
    # reset product for new neuron
        product = 0
    # get linear combination of weights and input
        for weight, input_ in zip(weights,inputs):
            product += weight * input_
        # add it to the output list
        outputs.append(product+bias)
    # check our output list
    print(outputs)

def batchNetwork():
    # create a batch of inputs
    inputs = [[1,5,7],[2,1,9],[3,3,4]]
    # create corresponding weights
    weights = [[.1,.2,.3],[.6,.5,.4],[.1,.2,.9]]
    # create corresponding bias
    biases = [5,2,2]

    # turn lists into numpy array
    inputsMatrix = np.array(inputs)
    weightsMatrix = np.array(weights)

    # use matrix multiplication + bias to get output matrix
    outputs = np.dot(inputsMatrix,weightsMatrix.T)+biases


class Dense_Layer:
    def __init__(self, number_of_inputs, number_of_neurons):
        #creates matrix of size (inputs,neurons) with random values
        self.weights = .01 * np.random.randn(number_of_inputs,number_of_neurons)
        #creates a (1,neurons) vector for biases
        self.biases = np.zeros((1,number_of_neurons))

    def foward(self, inputs):
        #calculates dot product of inputs and weights and adds biases
        self.output = np.dot(inputs,self.weights)+self.biases

#Testing our Dense Layer
import nnfs
from nnfs.datasets import spiral_data
#initializes random weights and zero biases
nnfs.init()
#gives us a random sample of data
X,y = spiral_data(100,3)
#Initalize dense layer with 2 inputs and 3 neurons
first_Layer = Dense_Layer(2,3)
#Calculate dot product + bias of random sample X and first_Layer
first_Layer.foward(X)

#grab a few outputs
print(first_Layer.output[:3])