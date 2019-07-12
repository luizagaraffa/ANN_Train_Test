#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 19:48:34 2018

Multilayer ANN training using Backpropagation and Strochastic Gradient Descent
Activation function: Sigmoid

@author: luizacgaraffa
"""


import numpy as np

"""
Function declaration
"""
# Definition of the sigmoidal activation function
def sigmoid(sum):
    return 1/(1+np.exp(-sum))

# Definition of the sigmoid function derivative for Errorr calculation using Gradient Descent
def sigmoidDerivative(sig):
    return sig*(1-sig)


"""
Variable declaration
"""
  
inputs = rna_in_70
outputs = rna_out_70


#inputs = np.array([[0.,0.,0.,0.,0.,0.],
#                     [0.,0.,0.,0.,0.,1.],  
#                    [0.,0.,0.,0.,1.,0.],
#                    [0.,0.,0.,0.,1.,1.]])
#outputs = np.array([[0.,0.,0.],[0.,0.,1.],[0.,0.,1.],[0.,0.,0.]])

nInputLayer = 6       # Number of neurons in the input layer
nHiddenLayer = 10     # Number of neurons in the hidden layer
nOutputLayer = 3      # Number of neurons in the output layer
nTrainingPairs = 996  # Number of training pairs in one epoch


# Randomly initialized weights
# Generate weights between -1/sqrt(nInputLayer) and +1/sqrt(nInputLayer) to avoid saturation

# Weights between input and hidden layer
Weights0 = np.random.uniform(low=-0.4,  high=0.4, size = (nInputLayer,nHiddenLayer))  
# Weights between hidden and output layer 
Weights1 = np.random.uniform(low=-0.4,  high=0.4, size = (nHiddenLayer, nOutputLayer)) # (linha, coluna); 2* e -1 para mesclar entre valores pos e neg

vector_MSE = []  
vector_MSE_mean = []  
sum_MSE = 0 
mean_MSE = 1                   


# Learning Rate: Defines how fast the algorithm will learn.
# Big: Fast convergence but you can lose the global minimum. 
# Small: Convergence slows down but is more likely to reach the global minimum
LearningRate = 1

# Momentum: Trys to escape from local minimum (not always works). Defines how reliable is the last change.
# Big: Increases the speed of convergence. Small: Can avoid local min
Momentum = 1

"""
Training Loop
"""
sumError = 0
MSE = 1
Epochs = 0

#while mean_MSE > 0.05:
for t in range(Epochs):
   Epochs = Epochs + 1
   
   for t in range(nTrainingPairs):
        MSE = 0
        InputLayer = inputs 

        sumSinapse0 = np.dot(InputLayer[t][:], Weights0) # Weighted sum on each hidden neuron
                                                
        HiddenLayer = sigmoid(sumSinapse0)               # Applys all weighted sums in the activation function
    
        sumSinapse1 = np.dot(HiddenLayer, Weights1)      # Weighted sum on each output neuron

        OutputLayer = sigmoid(sumSinapse1)               # Applys all weighted sums in the activation function
    
    # Start of the Weights update using Backpropagation. Steps: calculate delta of output and hidden
    # and apply those values on the weight update equation

        ErrorOutputLayer = outputs[t] - OutputLayer
        

    #Derivative of output layer for gradient descent calculation
        derivativeoutput = sigmoidDerivative(OutputLayer)  
    #Output Delta = sigmoid derivative * Error
        Deltaoutput = derivativeoutput * ErrorOutputLayer

    #Hidden layer Delta = sigmoid derivative * weight between output and hidden * output delta
        deltaHiddenLayer = np.zeros(nHiddenLayer)
        derivativeHiddenLayer = sigmoidDerivative(HiddenLayer)
        aux = []
        for i in range(nHiddenLayer):
            for j in range(nOutputLayer):
                aux.append(Deltaoutput[j] * Weights1[i,j] * derivativeHiddenLayer[i])
                deltaHiddenLayer[i] = sum(aux)
            aux = []

    # Hidden layer weights update
    # Weight(n+1) = Weight(n) * Momentum + Input * Delta * Learning Rate
    # HiddenLayerTranspose = HiddenLayer.T
    # WeightsNew1 =  np.array([HiddenLayer * (Deltaoutput)])  
        Weights1XMomentum = Weights1 * Momentum
        #deltaWeight1 = HiddenLayer * Deltaoutput * LearningRate
        for i in range(nHiddenLayer):
            for j in range(nOutputLayer):
                Weights1[i,j] = Weights1[i,j] + HiddenLayer[i] * Deltaoutput [j] * LearningRate
        
     # Input layer weights update
     # Weight(n+1) = Weight(n) * Momentum + Input * Delta * Learning Rate
        Weights0XMomentum = Weights0 * Momentum
        inputVector = np.squeeze(np.asarray(InputLayer[t][:]))
        for i in range(nInputLayer):
            for j in range(nHiddenLayer):        
                Weights0[i,j] = Weights0[i,j] + inputVector[i] * deltaHiddenLayer[j] * LearningRate
               
  
        MSE =  sum(ErrorOutputLayer**2)/nOutputLayer
        vector_MSE.append(MSE) 
        sum_MSE = sum_MSE + MSE
       
   
   mean_MSE = sum_MSE / nTrainingPairs #mean dos valores de MSE de uma epoca
   vector_MSE_mean.append(mean_MSE) 
   sum_MSE = 0 
   

   
   
print ("Total number of epochs: " + str(Epochs))



    
     