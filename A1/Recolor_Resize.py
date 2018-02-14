
# coding: utf-8

# In[6]:

import matplotlib 
#get_ipython().magic(u'matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt
import os
import scipy.misc

import time


# In[7]:

pic_list1 = os.listdir("cropped/") 
pic_list2 = os.listdir("cropped2/")


# In[8]:

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.


# In[9]:

start = time.time()
for pic in pic_list1:
    #print pic
    try:
        picture = matplotlib.image.imread("cropped/"+pic)
        picture = rgb2gray(picture)
        picture = scipy.misc.imresize(picture, (32, 32))
        scipy.misc.imsave("processed/"+pic, picture)
    except Exception as e:
        print(e)
        
    
    #plt.imshow(picture)
    #break
end = time.time()
print ("Time taken: %f" % (end - start))

start = time.time()
for pic in pic_list2:
    #print pic
    try:
        picture = matplotlib.image.imread("cropped2/"+pic)
        picture = rgb2gray(picture)
        picture = scipy.misc.imresize(picture, (32, 32))
        scipy.misc.imsave("processed2/"+pic, picture)
    except Exception as e:
        print(e)
        
    
    #plt.imshow(picture)
    #break
end = time.time()
print ("Time taken: %f" % (end - start))


# In[ ]:




# In[ ]:




# In[ ]:



