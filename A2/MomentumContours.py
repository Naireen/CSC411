
# coding: utf-8

# THis is is part 5, when creating the learning curves for part5, use python3

# In[1]:

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
import pickle
import os
from scipy.io import loadmat
# Load the MNIST digit data
M = loadmat("mnist_all.mat")


# In[2]:

def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y) / tile(sum(exp(y), 0), (len(y), 1))


# In[3]:

def tanh_layer(y, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y) + b)


# In[4]:

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output


# In[5]:

def cost_function(y, p):
    return -np.sum(y * np.log(p))


# In[6]:

def get_data(M):
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
        #print(i, data_size)
        train_data[data_counter: data_counter + data_size, :] = M[train_k]
        train_y[data_counter: data_counter + data_size, i] = 1
        data_counter += data_size
    return train_data, train_y

#plot learning curves
#collect testing data
def get_test_data(M):
    train_keys = [key for key in M.keys() if "test" in key]
    data_length = 0
    for train_k in train_keys:
        data_length += M[train_k].shape[0]

    # concatenate all the data
    train_data = np.zeros((data_length, 784))
    print(data_length)
    data_counter = 0
    data_size = 0

    train_y = np.zeros((data_length, 10))
    for i, train_k in enumerate(train_keys):
        data_size = M[train_k].shape[0]
        #print(i, data_size)
        train_data[data_counter: data_counter + data_size, :] = M[train_k]
        train_y[data_counter: data_counter + data_size, i] = 1
        data_counter += data_size
    return train_data, train_y


# In[7]:

def part2(x, W0, b0):
    total_W0 = np.concatenate((b0, W0))
    added_ones = np.ones(x.shape[1]).reshape(x.shape[1], 1)
    total_x = np.concatenate((added_ones.T, x))
    vals = softmax(np.dot(total_W0.T, total_x))
    return vals.T

def part3(x, y1, p):
    deriv = np.subtract(p, y1)
    added_ones = np.ones(x.shape[1]).reshape(x.shape[1], 1)
    total_x = np.concatenate((added_ones.T, x))
    deriv = np.matmul(deriv.T, total_x.T).T
    return deriv  # shape 10, 785


# In[8]:

def df(x, W0, y):
    # this W0 is assumed to have the bias incorperated into it
    #print(W0.shape, y.shape)
    W1, b0 = W0[:-1, :].reshape(784, W0.shape[1]), W0[-1, :].reshape(1, 10)
    vals = part2(x, W1, b0)
    return part3(x, y, vals)


# In[9]:

# PART ONE DATA SNIPPIT
# Load sample weights for the multilayer neural network
snapshot = pickle.load(open("snapshot50.pkl", "rb"), encoding="latin-1")
W0 = snapshot["W0"]  # data
b0 = snapshot["b0"].reshape((300, 1))  # bias?
W1 = snapshot["W1"]  # should be likelihoods?
b1 = snapshot["b1"].reshape((10, 1))  # classes?
#print(W0.shape)

np.random.seed(0)
weights = np.random.random(size=784*10).reshape((784, 10))/10.
weights_bias = np.zeros(10).reshape(1, 10)
#np.random.random(size=10).reshape((1, 10))



# In[13]:

def momemtum_grad_descent_learning_curves(df, x, y, W0, b0, alpha, testX, testY, momemtum = 0.9, EPS=1e-7):
    # EPS = 1e-5   #EPS = 10**(-5)
    start = time.time()
    total_W0 = np.concatenate((b0, W0))
    prev_totalW0 = total_W0 - 10 * EPS
    W1 = total_W0.copy()
    # inital weights, pass in both bias and weights together for now
    max_iter = 6500
    iter = 0
    performance_test = np.zeros(max_iter//100 +1) #check performance every 100 iterations
    performance_train = np.zeros_like(performance_test)

    added_ones = np.ones((testX.shape[1], 1))
    totaltestX = np.concatenate((testX.T,added_ones), axis=1)
    
    added_ones = np.ones((x.shape[1], 1))
    totaltrainX = np.concatenate((x.T, added_ones), axis = 1)
    
    counter = 1
    new_momentum = 1
    while norm(W1 - prev_totalW0) > EPS and iter < max_iter:
        prev_totalW0 = W1.copy()
        new_momentum = momemtum*new_momentum + alpha* df(x, W1, y)
        W1 -= new_momentum
        if iter % 100 == 0:
            print("Iter", iter)
            preds = np.matmul(totaltrainX, W1)
            preds_max =  preds.max(axis = 1)
            final_preds = np.isin(preds, preds_max).astype(int)
            #how many of the hot keys match for validation
            performance_train[counter] =  len(np.where((trainY == final_preds).all(axis =1)==True)[0])/60000.
            
            preds = np.matmul(totaltestX, W1)
            preds_max =  preds.max(axis = 1)
            final_preds = np.isin(preds, preds_max).astype(int)
            #how many of the hot keys match for validation
            performance_test[counter] = len(np.where((testY == final_preds).all(axis =1)==True)[0])  /10000.
            
            #counter +=1
            print(performance_train[counter], performance_test[counter])
            counter +=1
        iter += 1
        
    print("Iter", iter)
    end = time.time()
    print("Time taken:", (end - start))
    return W1, performance_train, performance_test


def part5(data, W0, b0, y, testX, testY):
    np.random.seed(0)
    alpha = 0.01
    final_weights, momt_train, momt_test = momemtum_grad_descent_learning_curves(df, data, y, W0, b0, alpha, testX, testY, 0.9,EPS=1e-4)
    return final_weights, momt_train, momt_test


# In[14]:

##PART FOUR##
trainX, trainY = get_data(M)
print(np.where(trainX==np.nan))
trainX = trainX.T/255. /255.
print("TrainX", trainX.shape)

testX, testY = get_test_data(M)
testX = np.nan_to_num(testX)
testX = testX.T/255./255.
print("TestX", testX.shape)

#final_weights = part5(trainX, weights, weights_bias, trainY, testX, testY)


# Plot the learning curves with these set of parameters every 400 iterations, see how the performance changes

# In[15]:

final_weights, performance_train, performance_test = part5(trainX, weights, weights_bias, trainY, testX, testY)


# In[17]:

print(performance_train, performance_test)
np.savetxt("performance_train_p5.txt", performance_train)
np.savetxt("performance_test_p5.txt", performance_test)


# In[34]:

matplotlib.rcParams.update({'font.size': 20})
plt.figure(figsize=(12 ,8))
iterations = np.arange(6500//100 +1)*100
#print(iterations)
#print(iterations.shape)
plt.plot(iterations, performance_train, label= "Training scores", marker = ".")
plt.plot(iterations, performance_test, label = "Testing scores", marker = ".")
plt.title("Performance with Gradient Descent with Momentum")
plt.xlabel("Number of Iterations")
plt.ylim([0, 1.])
plt.legend()
plt.show()

plt.figure(figsize=(12 ,8))
iterations = np.arange(6500//100 +1)*100
#print(iterations)
#print(iterations.shape)
plt.plot(iterations, performance_train, label= "Training scores", marker = ".")
plt.plot(iterations, performance_test, label = "Testing scores", marker = ".")
plt.title("Performance with Gradient Descent with Momentum")
plt.xlabel("Number of Iterations")
plt.ylim([0.8, .95])
plt.legend()
plt.show()


# In[36]:

#save weights so they can be used again without rerunningn every thing
np.savetxt("final_weightspart5.txt", final_weights)


# In[30]:

#part 5, gradient descent with momentum
#create a plot comparing this with vanilla gradient descent
p4_train = np.loadtxt("performance_train_p4.txt")
p4_test = np.loadtxt("performance_test_p4.txt")


# In[35]:

plt.figure(figsize=(12 ,8))
iterations = np.arange(6500//100 +1)*100
#print(iterations)
#print(iterations.shape)
plt.plot(iterations, performance_train, label= "Momentum Training scores", marker = ".")
plt.plot(iterations, performance_test, label = "Momentum Testing scores", marker = ".")
plt.plot(iterations, p4_train, label= "Vanilla Training scores", marker = ".")
plt.plot(iterations, p4_test, label = "Vanilla Testing scores", marker = ".")
plt.title("Comparing Vanilla and Momentum Gradient Descents")
plt.xlabel("Number of Iterations")
plt.ylim([0.8, .95])
plt.legend()
plt.show()


# In[ ]:



