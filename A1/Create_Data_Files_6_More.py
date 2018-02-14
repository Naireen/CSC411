
# coding: utf-8

# In[26]:

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from numpy.linalg import norm
import matplotlib.image as mpimg
import os


# In[27]:

#order txt files are produced for exmaple by ls radcliff* > Order4.txt # in the directory proccessed2.
list1 = np.loadtxt("processed2/Order.txt", dtype=str)[:10]
list2 = np.loadtxt("processed2/Order2.txt", dtype=str)[:10]
list3 = np.loadtxt("processed2/Order3.txt", dtype=str)[:10]
list4 = np.loadtxt("processed2/Order4.txt", dtype=str)[:10]
list5 = np.loadtxt("processed2/Order5.txt", dtype=str)[:10]
list6 = np.loadtxt("processed2/Order6.txt", dtype=str)[:10]
total_list = np.concatenate([list1, list2, list3, list4, list5, list6])
print total_list


# In[29]:

#train_data = np.zeros((220, 32 * 32))
total_data = np.zeros((60, 32**2))
for i, name in enumerate(total_list):
    #print pic.shape
    pic = mpimg.imread("processed2/" + name)
    #print pic.shape
    total_data[i, :] = pic[:, :].flatten()
total_data


# In[31]:

np.savetxt("new_6_test_data.txt", total_data)

