from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib


act = list(set([a.split("\t")[0].replace("\n", "") for a in open("subset_actors.txt").readlines()]))
print(act)




def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result
    
testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

for a in act:
    name = a.split()[1].lower()
    i = 0
    info = open("faces_subset.txt")
    for j, line in enumerate(info):
        #print line.split("\t")[0]

        if a in str(line.split("\t")[0]):
            #print("Is it here?")
            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #print line.split()[4]
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], "uncropped2/"+filename), {}, 45)
            #dimensions= line.split()[5].split(",")
            #print (i, filename, dimensions)
            #pic =  mpimg.imread("uncropped/"+filename)
            #print (i, filename, pic.shape, dimensions)
            try:
                dimensions= line.split()[5].split(",")
                pic =  mpimg.imread("uncropped2/"+filename)
                print(i, filename, pic.shape, dimensions)
                pic = pic[int(dimensions[1]):int(dimensions[3]), int(dimensions[0]):int(dimensions[2]), : ]
                imsave("cropped2/"+filename, pic)
            except Exception as e:
                print(e)
            #imsave("cropped/"+filename, pic)
            if not os.path.isfile("uncropped2/"+filename):
                continue

         
            #print filename
            i += 1
            #print (i)
            #break
        #if j ==5:
        #    break
    
    
