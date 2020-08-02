#!/usr/bin/env python3

import numpy as np
import copy
import sys
import pandas as pd
import pickle

def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def lrelu(x, alpha=0.01):
    a = np.where(x < 0, alpha * x, x)
    return a


def dlrelu(x, alpha=0.01):
    a = np.ones_like(x)
    a[x < 0] = alpha
    return a


def sigmoid_derivative(p):
    return p * (1 - p)


activation = {}
activation_derivative = {}
activation['sigmoid'] = sigmoid
activation_derivative['sigmoid'] = sigmoid_derivative
debug = False

class NeuralNetwork:
    def __init__(self, X, y, arch):
        self.arch = arch
        self.input = X[0]
        self.input = np.reshape(self.input, (-1, 1))
        self.weights = []
        self.layers = [None] * (len(self.arch))
        self.errors = copy.deepcopy(self.layers)
        self.Y = y
        self.X = X
        self.output = np.zeros(y[0].shape)
        self.label = np.zeros(y[0].shape)
        self.loss = float('inf')

        for i, item in enumerate(self.arch):
            if i == 0:
                self.weights.append(np.random.rand(self.input.shape[0], item['neurons']))
            else:
                if debug:
                    print(i, "SHAPE",self.weights[i-1].shape)
                self.weights.append(np.random.rand(self.weights[i - 1].shape[1], item['neurons']))

        if debug:
            print("LAST_SHAPE", self.weights[-1].shape)

        self.d_weights = copy.deepcopy(self.weights)

    def feedforward(self):
        if debug:
           print("FEED FORWARD")
        for i, layer in enumerate(self.arch):
            if i == 0:
                self.layers[i] = activation[layer['activation']](np.dot(self.input.T , self.weights[i]))
                if debug:
                    print(i, self.input.shape, ".", self.weights[i].shape, "->", self.layers[i].shape)

            else:
                prev_layer = self.layers[i - 1]
                curr_weights = self.weights[i]
                a = activation[layer['activation']]( np.dot(prev_layer, curr_weights))
                self.layers[i] = a
                if debug:
                    print(i, prev_layer.shape, ".", curr_weights.shape, "->", a.shape)

            if debug:
                print('layer', i, self.layers[i].shape)

        return self.layers[-1]

    def backprop(self):
        if debug:
            print("BACKPROP")
        for i, item in reversed(list(enumerate(self.layers))):
            if item is self.layers[-1]:
                self.errors[i] = self.label - item
                if debug:
                    print("OP ERROR",self.label.shape, item.shape)
                self.d_weights[i] = self.errors[i] * activation_derivative[self.arch[i]['activation']](item)
            elif item is self.layers[0]:
                self.errors[i] = np.dot(self.d_weights[i + 1] , self.weights[i + 1].T)
                self.d_weights[i] = np.dot(self.input , (
                            self.errors[i] * activation_derivative[self.arch[i]['activation']](item)))
            else:
                self.errors[i] = np.dot(self.d_weights[i + 1] , self.weights[i + 1].T)
                self.d_weights[i] = self.errors[i] * activation_derivative[self.arch[i]['activation']](item)
            if debug:
                print('error', i, self.errors[i].shape, 'd_weights', i, self.d_weights[i].shape)

        for i, item in enumerate(self.d_weights):
            if debug:
                print("UPDATE",i, self.weights[i].shape, item.shape)
            self.weights[i] += item

    def train(self):
        for i, item in enumerate(self.X):
            self.input = np.reshape(item, (-1,1))
            self.label = np.asarray([self.Y[i]])
            self.output = self.feedforward()
            self.backprop()
            self.loss = np.mean(np.sum((self.output - self.label) ** 2))


def one_hot_encode(Y):
    d = {
        0: [1,0,0,0],
        90: [0,1,0,0],
        180: [0,0,1,0],
        270: [0,0,0,1]
    }
    encoded_Y = np.asarray([d[i] for i in Y], dtype='float')
    return encoded_Y

def one_hot_decode(Y):

    decoded_Y = []
    d = {
        0: 0,
        1: 90,
        2: 180,
        3: 270
    }
    for y in Y:
        idx = np.argmax(y)
        decoded_Y.append(d[idx])


    return np.asarray(decoded_Y, dtype='int')

def train_nn(mode_file,model_file):
    data = pd.read_csv(mode_file, sep=" ", header=None)
    train_data = data.values[:, 1:]
    X = np.asarray(train_data[:, 1:], dtype='float')
    Y = train_data[:, 0]
    y = one_hot_encode(Y)


    arch = [
        {'type': 'hidden', 'neurons': 70, 'activation': 'sigmoid'},
        {'type': 'hidden', 'neurons': 40, 'activation': 'sigmoid'},
        {'type': 'hidden', 'neurons': 20, 'activation': 'sigmoid'},
        {'type': 'output', 'neurons': 4, 'activation': 'sigmoid'},

    ]
    NN = NeuralNetwork(X, y, arch)
    for i in range(10):
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(NN.input))
        print("Actual Output: \n" + str(NN.label))
        print("Predicted Output: \n" + str(NN.output))
        print("Iteration: " + str(i), "Loss: " + str(NN.loss))  # mean sum squared loss
        print("\n")

        NN.train()
    with open(model_file,'wb') as file:
        pickle.dump(NN,file)


def test_nn(mode_file, model_file):
    data = pd.read_csv(mode_file, sep=" ", header=None)
    train_data = data.values[:, 1:]
    X = np.asarray(train_data[:, 1:], dtype='float')
    Y = train_data[:, 0]
    y = one_hot_encode(Y)

    with open(model_file,'rb') as file:
        NN=pickle.load(file)

    total = X.shape[0]
    n_correct = 0
    pred_y = []
    for i,item in enumerate(X):
        pred = NN.feedforward(item)
        if y[i] == pred:
            n_correct+=1
        pred_y.append(pred)

    print("Accuracy of NN is", n_correct*100/total, "%")

    pred_y = one_hot_decode(pred_y)
    with open("nn_output.txt", 'w') as out:
        for i, item in enumerate(pred_y):
            out.write( data[i][0] +" "+ str(item))





if __name__=='__main__':

    mode_file = 'train_file.txt'

    data = pd.read_csv(mode_file, sep=" ", header=None)
    train_data = data.values[:, 1:]
    X = np.asarray(train_data[:, 1:], dtype='float')
    Y = train_data[:, 0]
    y = one_hot_encode(Y)

    debug = False

    arch = [
        {'type': 'hidden',  'neurons': 70, 'activation': 'sigmoid'},
        {'type': 'hidden', 'neurons': 40, 'activation': 'sigmoid'},
        {'type': 'hidden', 'neurons': 20, 'activation': 'sigmoid'},
        {'type': 'output', 'neurons': 4, 'activation': 'sigmoid'},

    ]
    NN = NeuralNetwork(X, y, arch)
    for i in range(10):
            print("for iteration # " + str(i) + "\n")
            print("Input : \n" + str(NN.input))
            print("Actual Output: \n" + str(NN.label))
            print("Predicted Output: \n" + str(NN.output))
            print("Iteration: " + str(i), "Loss: " + str(NN.loss))  # mean sum squared loss
            print("\n")

            NN.train()
