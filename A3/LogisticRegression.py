
# coding: utf-8

# In[1]:

import numpy as np
import pickle
from numpy.linalg import norm


# In[2]:

import os
print(os.listdir("."))
train_r = pickle.load(open("train_r.pkl", "rb"))
train_b = pickle.load(open("train_b.pkl", "rb"))
valid_r = pickle.load(open("valid_r.pkl", "rb"))
valid_b = pickle.load(open("valid_b.pkl", "rb"))


# In[3]:

from sklearn.linear_model import LogisticRegression


# In[4]:

lr = LogisticRegression()


# In[5]:

print(valid_r[:3])


# In[6]:

one_label = np.ones(len(train_r))
zeros_label = np.zeros(len(train_b))
one_test = np.ones(len(valid_r))
zero_test = np.zeros(len(valid_b))
trainX = np.concatenate((train_r, train_b))
trainY = np.concatenate((one_label, zeros_label))
validX = np.concatenate((valid_r, valid_b))
validY  = np.concatenate((one_test, zero_test))
print(validX[:3], len(validX))


# In[7]:

#gerenate feature vectors 
total_words = {}
for line in trainX:
    words = line.split(" ")
    for word in words:
        if word == "":
            continue
        try:
            total_words[word]
        except:
            total_words[word] = True

print(len(total_words.keys()))
#dont add words in valid and test sets


# In[8]:

#create training feats

#print(len(validX))
unique_keys = list(total_words.keys())
unique_keys.sort()
#print(unique_keys[:10])
#account for a bias
train_feats = np.zeros((len(trainX), len(total_words.keys()) ))
for i, line in enumerate(trainX):
    words = line.split(" ")
    #print(words)
    for word in words:
        if word =="":
            continue
        #print(unique_keys.index(word))
        ind = unique_keys.index(word)  
        #plus one is so that the very first is the bias (here a bias of zero)
        train_feats[i][ind] = 1


#create testing features
valid_feats = np.zeros((len(validX), len(total_words.keys())))
for i , line in enumerate(validX):
    words = line.split(" ")
    #print(len(words), end = " ")
    #if i ==0:
        #print(words)
    #print(words)
    for word in words:
        if word =="":
            continue
        #account for value error:
        try:
            ind = unique_keys.index(word) 
            #print(ind)
            valid_feats[i][ind] = 1
        except:
            continue       
#inds = np.where(valid_feats ==1)[1]
#print("\n", inds[:34])


# In[46]:

#logistic regression model from assignment 1.
def grad_descent(f, df, x, y, reg, init_t, alpha, EPS=1e-4):
    print(reg)
    #EPS = 1e-4   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 300
    iter  = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t, reg)
        if iter % 50 == 0:
            print("Iter", iter)
        iter += 1
    print("Iter", iter)
    return t


# In[75]:

def softmax(x):
    return 1./(1+np.exp(-1*x))
#assume here that x is already flattened
def f_log(x, y, theta, reg):
    bias = np.ones( (1, x.shape[0])).T
    x = np.vstack( (bias.T, x.T))
    val = softmax(np.dot(theta.T, x))
    vals =  np.sum((y - np.log(val)) - (1.-y)*np.log(1-val)) + reg*np.sum(theta**2)
    return vals
    
def df_log(x, y, theta ,reg):
    bias = np.ones( (1, x.shape[0])).T
    x = np.vstack( (bias.T, x.T))
    val = softmax(np.dot(theta.T, x)) - y
    val = np.dot(val, x.T)
    val += 2.*reg*theta
    return val
    #return   -2*np.sum((y-np.dot(theta.T, x))*x, 1) + 2.*reg*theta


# In[93]:

#assume here that x is already flattened
def f_l1(x, y, theta, reg):
    bias = np.ones( (1, x.shape[0])).T
    x = np.vstack( (bias.T, x.T))
    val = softmax(np.dot(theta.T, x))
    vals =  np.sum((y - np.log(val)) - (1.-y)*np.log(1-val)) + reg*np.sum(theta)
    return vals 

def df_l1(x, y, theta ,reg):
    bias = np.ones( (1, x.shape[0])).T
    x = np.vstack( (bias.T, x.T))
    val = softmax(np.dot(theta.T, x)) - y
    val = np.dot(val, x.T)
    val += reg
    return val


# In[76]:

np.random.seed(0)
#this is used to create the two differentn types of images, uncomment the one that you dont want
theta0 = np.random.random((len(unique_keys)+1))*5
theta0 = np.ones_like(theta0)


# In[77]:

theta = grad_descent(f_log, df_log, train_feats, trainY,0.15,  theta0, 0.0005, EPS=5e-4)


# In[78]:

#learning curve
valid_data_ones = np.concatenate((np.ones((1, valid_feats.shape[0])).T, valid_feats), axis = 1)
preds = np.dot( valid_data_ones, theta)
print(valid_data_ones.shape, theta.shape, preds.shape)
preds_binary= np.zeros_like(preds)
#preds_binary[preds<0.5] = 0
preds_binary[preds>0.5] = 1
#print(preds)
print(len(np.where(validY ==preds_binary)[0]), validY.shape[0])
print(len(np.where(validY ==preds_binary)[0])/ validY.shape[0])
#print(validY)


# In[91]:

#l2error function

#logistic regression model from assignment 1.
def track_grad_descent(f, df, x, y,reg, init_t, alpha, EPS=1e-4):
    #EPS = 1e-4   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    #times = np.linspace()
    max_iter = 2500
    iter  = 0
    thresh = 25
    iterations = np.arange(max_iter//thresh +1)*thresh
    learn_train = [0]#np.zeros_like(iterations)
    learn_valid = [0]#np.zeros_like(iterations)
    counter = 1
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t, reg)
        if iter % 500 == 0:
            print("Iter", iter)
        if (iter%thresh==thresh-1):
            #train scores
            train_data_ones = np.concatenate((np.ones((1, train_feats.shape[0])).T, train_feats), axis = 1)
            preds = np.dot(train_data_ones, prev_t)
            preds_binary= np.zeros_like(preds)
            preds_binary[preds<0.5] = 0
            preds_binary[preds>0.5] = 1
            val = len(np.where(trainY ==preds_binary)[0])/ float(trainY.shape[0])
            print(val)

            learn_train.append(val)

            valid_data_ones = np.concatenate((np.ones((1, valid_feats.shape[0])).T, valid_feats), axis = 1)
            preds = np.dot( valid_data_ones, prev_t)
            preds_binary= np.zeros_like(preds)
            preds_binary[preds<0.5] = 0
            preds_binary[preds>0.5] = 1
            val =len(np.where(validY ==preds_binary)[0])/ float(validY.shape[0])
            print(val)
            learn_valid.append(val)
            #learn_valid[counter] = val
            #print(learn_valid)
            #counter +=1
            
        iter += 1
    print(learn_train, learn_valid)
    print("Iter", iter)
    return t, learn_train, learn_valid


# In[94]:

#%%time
#done with l2 regularization
np.random.seed(0)
theta0 = np.random.random((len(unique_keys)+1))
theta, train_learn, valid_learn = track_grad_descent(f_log, df_log, train_feats, trainY, 5, theta0, 0.0005, EPS=5e-4)


# In[98]:

import matplotlib
from matplotlib import pyplot as plt
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 20}
#matplotlib.rc('text', usetex=True)
matplotlib.rc('font', **font)


# In[101]:

iterations = np.arange(1176//25 +1)*25


plt.figure(figsize = (10, 7))
plt.plot(iterations, train_learn, marker = ".", label = "Training Accuracy")
plt.plot(iterations, valid_learn, marker = ".", label = "Testing Accuracy")

plt.legend()
plt.show()
#print(train_learn)


# In[102]:

#%%time
#done with l1 regularization
np.random.seed(0)
theta0 = np.random.random((len(unique_keys)+1))
theta, train_learn, valid_learn = track_grad_descent(f_l1, df_l1, train_feats, trainY, 5, theta0, 0.0005, EPS=5e-4)


# In[103]:

iterations = np.arange(2500//25 +1)*25


plt.figure(figsize = (10, 7))
plt.plot(iterations, train_learn, marker = ".", label = "Training Accuracy")
plt.plot(iterations, valid_learn, marker = ".", label = "Testing Accuracy")

plt.legend()
plt.show()
#print(train_learn)


# #write up part 5

# In[ ]:




# In[ ]:




# Its reasonable to use magnitude for this, since the different scale for the features have meaning, if one is larger, it means that it occured more often(native bayes method)
# For this one, the individual features are already normalized, since they're either zero if that word is not present, or are one if they are. 

# In[ ]:



