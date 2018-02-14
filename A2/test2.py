from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random
import sys

# reload(sys)

# sys.setdefaultencoding("utf8")
import pickle

import os
from scipy.io import loadmat

# Load the MNIST digit data
M = loadmat("mnist_all.mat")


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y) / tile(sum(exp(y), 0), (len(y), 1))


def tanh_layer(y, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y) + b)


def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output


def cost_function(y, p):
    return -np.sum(y * np.log(p))


def part2(x, W0, b0):
    total_W0 = np.concatenate((b0, W0))
    added_ones = np.ones(x.shape[1]).reshape(x.shape[1], 1)
    #print(x.shape, added_ones.shape)
    total_x = np.concatenate((added_ones.T, x))
    vals = softmax(np.dot(total_W0.T, total_x))
    print("VAls", vals.shape)
    return vals


def part3(x, y1, p):
    print(y1.shape, p.shape)
    deriv = np.subtract(p, y1.T)
    added_ones = np.ones(x.shape[1]).reshape(x.shape[1], 1)
    total_x = np.concatenate((added_ones.T, x))
    deriv = np.matmul(deriv, total_x.T).T
    return deriv  # shape 10, 785


def get_data(M):
    # M is from the loaded MAT file
    train_keys = [key for key in M.keys() if "train" in key]
    data_length = 0
    for train_k in train_keys:
        data_length += M[train_k].shape[0]

    # concatenate all the data
    train_data = np.zeros((data_length, 784))
    data_counter = 0
    data_size = 0

    train_y = np.zeros((data_length, 10))
    for i, train_k in enumerate(train_keys):
        data_size = M[train_k].shape[0]
        # print(train_data[data_counter:data_counter+])
        train_data[data_counter: data_counter + data_size, :] = M[train_k]
        train_y[data_counter: data_counter + data_size, i] = 1
        data_counter += data_size
    return train_data, train_y


def df(x, W0, y):
    # this W0 is assumed to have the bias incorperated into it
    W1, b0 = W0[:-1, :], W0[-1, :]
    vals = part2(x, W1, b0)
    return part3(x, y, vals)


def grad_descent(df, x, y, W0, b0, alpha, EPS=1e-5):
    # EPS = 1e-5   #EPS = 10**(-5)
    total_W0 = np.concatenate((b0, W0))
    print(b0.shape)
    prev_totalW0 = total_W0 - 10 * EPS
    # prev_b0 = b0 - 10 *EPS
    W0 = total_W0.copy()
    # inital weights, pass in both bias and weights together for now
    max_iter = 10000
    iter = 0
    while norm(W0 - prev_totalW0) > EPS and iter < max_iter:
        prev_totalW0 = W0.copy()
        # prev_b0 = b0.copy()
        # total_
        W0 -= alpha * df(x, W0, y)
        if iter % 500 == 0:
            print("Iter", iter)
            # print "x = (%.3f, %.3f, %.3f), f(x) = %.3f" % (t[0], t[1], t[2], f(x, y, t))
            # print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    print("Iter", iter)
    return t


def part4(data, W0, b0, y):
    np.random.seed(0)
    # let W0 represent the initial weights
    weights = grad_descent()

    return


def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    p_i = part2(W0, b0, weights)  # the probabilities for each data for each class
    derivative = -(W1.T - p_i)
    d_W = np.matmul(derivative, W0)
    dCdL1 = y - y_
    dCdW1 = dot(L0, dCdL1.T)


if __name__ == "__main__":
    # PART ONE DATA SNIPPIT
    # Load sample weights for the multilayer neural network
    snapshot = pickle.load(open("snapshot50.pkl", "rb"), encoding="latin-1")
    W0 = snapshot["W0"]  # data
    b0 = snapshot["b0"].reshape((300, 1))  # bias?
    W1 = snapshot["W1"]  # should be likelihoods?
    b1 = snapshot["b1"].reshape((10, 1))  # classes?
    print(W0.shape)
    np.random.seed(0)
    weights = np.random.random(size=784 * 10).reshape((784, 10))
    weights_bias = np.random.random(size=10).reshape((1, 10))

    trainX, trainY = get_data(M)
    trainX = trainX.T
    trainX = trainX/225.
    print (trainX.shape)

    output = part2(trainX,weights, weights_bias)
    deriv = part3(trainX, trainY, output)
    print(deriv.shape)

    '''
    # PART TWO - SINGLE LAYER NN
    x = M["train5"][148:149].T
    x = x / 255.
    print(x.shape)
    output = part2(x, weights, weights_bias)
    print(output.shape)  # out puts probs
    y = np.zeros((10, 1))
    y[5, 0] = 1.
    # get this to work for more than one x at a time
    print(part3(x, y, output).shape)
    '''

    '''
    # PART TWO - SINGLE LAYER NN
    x = M["train5"][148:149].T / 255.
    output = part2(x, weights, weights_bias)
    y = np.zeros((10, 1))
    y[5, 0] = 1.

    # PART THREE _ FINITE DIFFS
    cost = cost_function(y, output)

    h = 0.0001
    for i in range(5):
        place = 215 + i
        weights2 = weights.copy()
        weights2[place, 5] += h
        output2 = part2(x, weights2, weights_bias)
        cost2 = cost_function(y, output2)
        print("Finite Diff", (cost2 - cost) / h)
        real_diff = part3(x, y, output)
        print("Real Diff", real_diff[place + 1, 5])
        # print(real_diff.shape)

    ##PART FOUR##
    trainX, trainY = get_data(M)
    #grad_descent(df, trainX, trainY, weights, weights_bias, 0.01)

    # L0, L1, output = forward(x, W0, b0, W1, b1)
    # get the index at which the output is the largest
    # y = argmax(output)
    '''
    ################################################################################
    # Code for displaying a feature from the weight matrix mW
    # fig = figure(1)
    # ax = fig.gca()
    # heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)
    # fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    # show()
    ################################################################################
