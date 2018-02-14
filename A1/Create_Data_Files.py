
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy.linalg import norm
import matplotlib.image as mpimg
import os


# In[21]:

list_pics = os.listdir("Train/")
print(list_pics)
alec_pics = [x for x in list_pics if "baldwin" in x]
steve_pics = [x for x in list_pics if "carell" in x]
a= np.zeros(len(alec_pics)) 
b= np.ones(len(steve_pics))
print a.shape, b.shape
test_y = np.concatenate([a, b])
train_y = np.concatenate( [a, b] )

test_pics = os.listdir("Test/")
alec_test_pics = [x for x in test_pics if "baldwin" in x]
steve_test_pics = [x for x in test_pics if "carell" in x]
a= np.zeros(len(alec_test_pics)) 
b= np.ones(len(steve_test_pics))
test_y = np.concatenate([a, b])


test_pics = os.listdir("Validation/")
alec_valid_pics = [x for x in test_pics if "baldwin" in x]
steve_valid_pics = [x for x in test_pics if "carell" in x]
a= np.zeros(len(alec_valid_pics)) 
b= np.ones(len(steve_valid_pics))
valid_y = np.concatenate([a, b])
#shuffle the test set?



# In[23]:

train_data = np.zeros((220, 32 * 32))

for i, name in enumerate(alec_pics):
    #print pic.shape
    pic = mpimg.imread("Train/" + name)
    train_data[i, :] = pic[:, :].flatten()
    #break
for i, name in enumerate(steve_pics):
    pic = mpimg.imread("Train/" + name)
    train_data[i+110, :] = pic[:,:].flatten()
#train_data


# In[24]:

np.savetxt("train_data.txt", train_data)


# In[25]:

test_data = np.zeros((20, 32 * 32))

for i, name in enumerate(alec_test_pics):
    #print pic.shape
    pic = mpimg.imread("Test/" + name)
    test_data[i, :] = pic[:, :].flatten()
    #break
for i, name in enumerate(steve_test_pics):
    pic = mpimg.imread("Test/" + name)
    test_data[i+10, :] = pic[:,:].flatten()
#test_data


# In[26]:

np.savetxt("test_data.txt", test_data)


# In[27]:

valid_data = np.zeros((20, 32 * 32))

for i, name in enumerate(alec_valid_pics):
    #print pic.shape
    pic = mpimg.imread("Validation/" + name)
    valid_data[i, :] = pic[:, :].flatten()
    #break
for i, name in enumerate(steve_valid_pics):
    pic = mpimg.imread("Validation/" + name)
    valid_data[i+10, :] = pic[:,:].flatten()
#valid_data


# In[28]:

np.savetxt("valid_data.txt", valid_data)


# In[29]:

trainy = np.concatenate((np.zeros(110), np.ones(110) ))
np.savetxt("train_y.txt", trainy)
testy = np.concatenate((np.zeros(10), np.ones(10) ))
np.savetxt("test_y.txt", testy)
np.savetxt("valid_y.txt", testy)


# In[30]:

# create data for entire dataset for the 6 actors


# In[31]:

file_order = np.loadtxt("Train/Order.txt", dtype=str)

train_data = np.zeros((660, 32 * 32))

for i, name in enumerate(file_order):
    #print pic.shape
    pic = mpimg.imread("Train/" + name)
    train_data[i, :] = pic[:, :].flatten()
train_data.shape
np.savetxt("train_data_6.txt", train_data)
#the data is ordered so its bladwin, bracco, carell, gilpin, hader, harmon
# M F M F M F
# zero is males, one is female
train_y = np.concatenate( (np.zeros(110), np.ones(110) ))
train_y = np.concatenate((train_y, train_y, train_y))
print train_y.shape
np.savetxt("train_y_6.txt", train_y)


# In[32]:

file_order = np.loadtxt("Validation//Order.txt", dtype=str)

train_data = np.zeros((60, 32 * 32))

for i, name in enumerate(file_order):
    #print pic.shape
    pic = mpimg.imread("Validation//" + name)
    train_data[i, :] = pic[:, :].flatten()
train_data.shape
np.savetxt("valid_data_6.txt", train_data)
#the data is ordered so its bladwin, bracco, carell, gilpin, hader, harmon
# M F M F M F
# zero is males, one is female
train_y = np.concatenate( (np.zeros(10), np.ones(10) ))
train_y = np.concatenate((train_y, train_y, train_y))
print train_y.shape
np.savetxt("valid_y_6.txt", train_y)


# In[ ]:



