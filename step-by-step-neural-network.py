# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 12:58:23 2017
inspired by :
    https://iamtrask.github.io/2015/07/12/basic-python-network/
    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    https://www.youtube.com/watch?v=ILsA4nyG7I0
    https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
@author: lrosique
"""
import numpy as np
# seed random numbers to make calculation deterministic (just a good practice)
np.random.seed(1)

#################################
#################################
# FUNCTIONS
# Sum of weights at layer level
def sum_weight(X, w):
    sums = []
    for i in range(np.size(w,0)):
        sums.append(sum_weight_on_neuron(X,w, i))
    return np.array(sums)

# Sum of weights at neuron level
def sum_weight_on_neuron(X, w, neuron_number):
    return np.sum(X*w[neuron_number])

# Function sigmoïd (positive)
def sigmoid_positive(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
sigmoid_positive = np.vectorize(sigmoid_positive)

# Function sigmoïd (between -1 and 1)
def sigmoid(x,deriv=False):
    if(deriv==True):
        return 2*sigmoid_positive(x, True)
    return 2*sigmoid_positive(x) - 1
sigmoid = np.vectorize(sigmoid)

# Function identity
def identity(x,deriv=False):
    if(deriv==True):
        return 1
    return x
identity = np.vectorize(identity)

# Function ReLU
def relu(x,deriv=False):
    if x > 0:
        if(deriv==True):
            return 1
        else:
            return x
    return 0
relu = np.vectorize(relu)

# Automatic generation of weights at layer level
def genererate_weights_layer(size_n1, size_n2):
    # Testing case : weights go by +1 at each connection
    wlayer = np.arange(size_n1*size_n2).reshape(size_n2, size_n1)
    # Testing case : all equal to 1
    # wlayer = np.ones(size_n1*size_n2).reshape(size_n2, size_n1)
    # Random case (usual)
    # wlayer = np.random.random((size_n2,size_n1))
    return wlayer

# Automatic generation of weights at network level
def genererate_weights_network(size_layer):
    size_n1 = None
    size_n2 = None
    weights_network = []
    for x in np.nditer(size_layer):
        size_n2 = x
        if(size_n1 is not None):
            weights_network.append(genererate_weights_layer(size_n1, size_n2))
        size_n1 = x
    return np.array(weights_network)

# Full run forward of a network
def evaluate_network(X_in, weight, nb_neurons_per_layer, functions_at_neurons):
    list_sums = []
    list_functions = []
    result_layer = X_in
    for i in range(np.size(nb_neurons_per_layer, 0) - 1):
        sum_weights_layer = sum_weight(result_layer, weight[i])
        result_layer = globals().get(functions_at_neurons[i])(sum_weights_layer)
        list_sums.append(sum_weights_layer)
        list_functions.append(result_layer)
    return [list_sums, list_functions]

# Squarred error per neuron on output
#def squared_error_per_neuron(expected, output):
#    return 1/2*(expected - output)**2

# Total squared error
#def squared_error_function(expected, output):
#    return np.sum(squared_error_per_neuron(expected, output))

# Calculate the error on each weight of the last layer
#def last_weights_error(output, expected_output, function_last_layer):
#    correction = []
#    for i  in range(np.size(networkWeights[-1:][0], 0)):
#        tmp = []
#        for j in range(np.size(networkWeights[-1:][0], 1)):
#            tmp.append(-(expected_output[i] - output[i])*output[i]*function_last_layer(output[i],True))
#        correction.append(tmp)
#    return correction

#################################
#################################
#### CONFIGURATION
# First layer is input and Last layer is output
nbNeuronsPerLayers = np.array([3, 4, 3, 2])

# Input of the networ, of size nbNeuronsPerLayers[0]
XInput = np.array([0, 1, 2])

# Activation functions, of size nbNeuronsPerLayers.size - 1
layersFunctions = np.array(['sigmoid','sigmoid','relu'])

# Weights generation
networkWeights = genererate_weights_network(nbNeuronsPerLayers)

# Expectation
YExpected = np.array([1, 14])

#################################
#################################
#### EXEMPLE Forward : step by step
# By hand
sum_w_layer1 = sum_weight(XInput, networkWeights[0])
function_layer1 = sigmoid(sum_w_layer1)

sum_w_layer2 = sum_weight(function_layer1, networkWeights[1])
function_layer2 = sigmoid(sum_w_layer2)

sum_w_output = sum_weight(function_layer2, networkWeights[2])
function_output = relu(sum_w_output)
output = function_output

# Automatic
list_round1 = evaluate_network(XInput, networkWeights, nbNeuronsPerLayers, layersFunctions)
output_round1 = list_round1[1][-1:][0]

#################################
#################################
#### EXEMPLE Backpropagation : step by step
# By hand
#output_error = squared_error_per_neuron(YExpected, output)
#o1_error = output_error[0]
#o2_error = output_error[1]
#total_error = np.sum(output_error)

#correction_last_layer = last_weights_error(output, YExpected, relu)

#correction_l2

#correction_l1