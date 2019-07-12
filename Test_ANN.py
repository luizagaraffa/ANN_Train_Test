#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:02:52 2018
Algorithm of ANN inference, for testing and validation

@author: luizacgaraffa

"""

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""
Function Declaration
"""
# Definition of the sigmoid activation function
def sigmoid(sum):
    return 1/(1+np.exp(-sum))

# Definition of the sigmoid function derivative for Errorrr calculation using Gradient Descent
def SigmoidDerivative(sig):
    return sig*(1-sig)


"""
Variable declaration
"""

Inputs = rna_in_30
Outputs = rna_out_30


#Inputs = np.array([[0.,0.,0.,0.,1.,0.],
#                    [0.,0.,0.,0.,1.,1.],  
#                    [0.,0.,0.,0.,1.,0.],
#                    [0.,0.,0.,0.,1.,1.],
#                    [0.,0.,0.,0.,1.,0.],
#                    [0.,0.,0.,0.,1.,1.],  
#                    [0.,0.,0.,0.,0.,0.],
#                    [0.,0.,0.,0.,1.,1.],
#                    [0.,0.,0.,0.,1.,0.],  
#                    [0.,0.,0.,0.,0.,1.]])
#Outputs = np.array([[0.,0.,1.],[0.,0.,0.],[0.,0.,1.],[0.,0.,0.],[0.,0.,1.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,1.],[0.,0.,1.]])



nInputLayer = 6       # Number of neurons in the input layer
nHiddenLayer = 10     # Number of neurons in the hidden layer
nOutputLayer = 3      # Number of neurons in the output layer
nTrainingPairs = 451  # Number of training pairs in one epoch


Weights0 = Weights0   # Weights between input and hidden layer

Weights1 = Weights1   # Weights between hidden and output layer

vector_MSE_treino = []
sum_MSE = 0
mean_MSE =0
Outputs_obtained = []

"""
Test loop
"""

#----------------Testing a database-----------------------------
# Comment to test one single vector
for g in range(nTestingPairs):
    MSE = 0
    layerInput = Inputs #Variavel auxiliar
    sumSinapse0 = np.dot(layerInput[g][:], Weights0) # Weighted sum in each hidden neuron
                                                 
    layerHidden = sigmoid(sumSinapse0)     	     # Applys all weighted sums in the activation function

    sumSinapse1 = np.dot(layerHidden, Weights1)      # Weighted sum on each output neuron

    layerOutpur = sigmoid(sumSinapse1)               # Applys all weighted sums in the activation function


    Outputs_obtained.append(layerOutpur)
    ErrorlayerOutpur = Outputs[g] - layerOutpur

    MSE =  sum(ErrorlayerOutpur**2)/ nOutputs
    print("MSE = " + str(MSE)) 
    vector_MSE_treino.append(MSE)
    sum_MSE = sum_MSE + MSE
mean_MSE = sum_MSE / nTestingPairs
print("mean MSE: " + str(mean_MSE))


# ------------------Testing one single vector----------------
#Comment to test a dataset

MSE = 0
layerInput = Inputs 
sumSinapse0 = np.dot(layerInput[50][:], Weights0) # Weighted sum in each hidden neuron
                                                
layerHidden = sigmoid(sumSinapse0)                # Applys all weighted sums in the activation function

sumSinapse1 = np.dot(layerHidden, Weights1)       # Weighted sum on each output neuron

layerOutpur = sigmoid(sumSinapse1)                # Applys all weighted sums in the activation function

ErrorlayerOutpur = Outputs[50] - layerOutpur

MSE =  sum(ErrorlayerOutpur**2)/ nOutputs
print("MSE = " + str(MSE))""" 

#----------------------------------------------------------------


#Confusion Matrix
y_pred = np.reshape(Outputs_obtained, ((nTestingPairs*nOutputs), 1))
#accuracy_score(Outputs, Outputs_obtained, normalize=False)
y_pred_binario = [round(x) for x in y_pred]
Outputs_obtained_binarias = np.reshape(y_pred_binario, (nTestingPairs, nOutputs))
cm = confusion_matrix(np.argmax(Outputs, axis=1), np.argmax(Outputs_obtained_binarias, axis=1))
#cm = confusion_matrix(Outputs, Outputs_obtained_binarias)

print(cm)

# Show confusion matrix in a separate window
#plt.matshow(cm)
#plt.title('Confusion matrix')
#plt.colorbar()
#plt.ylabel('True label')
#plt.xlabel('Predicted label')
#plt.show()