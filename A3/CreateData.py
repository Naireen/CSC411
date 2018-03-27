
# coding: utf-8

# In[1]:

#print(valid_r_preds[:, 1])
#calculate the predictions of it being real for all, fake is the oppostion
import matplotlib
from matplotlib import pyplot as plt
#final_guesses[:len(valid_r)]


# In[2]:

import numpy as np
import pandas as pd
import os
import time
import pickle


# In[3]:

#create data
#good = np.loadtxt("clean_real.txt", dtype = str)
def make_data():
    good = open("clean_real.txt", "r")
    words = good.read()
    words = words.split("\n")
    #print(len(words))
    total = len(words)
    np.random.seed(0)
    np.random.shuffle(words)
    #print(words[0])
    end1 = int(total*0.7)
    end2 = int(total*0.85)
    train = words[:end1]
    valid = words[end1:end2]
    test = words[end2:]
    print(len(train), len(valid), len(test))

    bad = open("clean_fake.txt", "r")
    words = bad.read()
    words = words.split("\n")
    #print(len(words))
    total = len(words)
    np.random.seed(0)
    np.random.shuffle(words)
    #print(words[0])
    end1 = int(total*0.7)
    end2 = int(total*0.85)
    train_bad = words[:end1]
    valid_bad = words[end1:end2]
    test_bad = words[end2:]
    print(len(train_bad), len(valid_bad), len(test_bad))
    
    return train, valid, test, train_bad, valid_bad, test_bad
p1, p2, p3, p4, p5, p6 = make_data()


# In[7]:

#p1.extend(p4)
#p2.extend(p5)
#p3.extend(p6)
pickle.dump(p1, open("train_r.pkl", "wb"))
pickle.dump(p2, open("valid_r.pkl", "wb"))
pickle.dump(p3, open("test_r.pkl", "wb"))
pickle.dump(p4, open("train_b.pkl", "wb"))
pickle.dump(p5, open("valid_b.pkl", "wb"))
pickle.dump(p6, open("test_b.pkl", "wb"))

p1 = np.asarray(p1)
p2 = np.asarray(p2)
p3 = np.asarray(p3)

p4 = np.asarray(p4)
p5 = np.asarray(p5)
p6 = np.asarray(p6)
real = np.concatenate((p1, p2, p3))
fake = np.concatenate((p4, p5, p6))


# Use the native Bayes Algorithm

# In[16]:

for line in fake[:40]:
    print(line)
    
print()
for line in real[:40]:
    print(line)


# Determine some stats to guess which three words will determine fake and real news
# 

# The key words donald and hilary are not useful, sinc ethey appear alot in real and fake news
# check obama

# In[43]:

real_words = real.flatten()
total_real = real.shape[0]
real_words = " ".join(real_words)
print(real_words[:140])

fake_words = fake.flatten()
total_fake = fake.shape[0]
fake_words = " ".join(fake_words)
print(fake_words[:140])
print(total_real, total_fake)


# determine stats

# In[44]:

#count words of trump in both list
import re


# In[57]:

def count_words(word, real_words, fake_words):
    count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), real_words))
    count2 = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(word), fake_words))
    #print("Word: {0}, Count in real: {1}/{3}, Count in fake : {2}/{4}".format(word, count, count2, total_real, total_fake))
    print(" {0} {1:.04} {2:.04}".format(word, count/total_real, count2/total_fake, total_real, total_fake))


# In[58]:

count_words("trump", real_words, fake_words)
count_words("obama", real_words, fake_words)
count_words("hillary", real_words, fake_words)
count_words("immigration", real_words, fake_words)
count_words("muslim", real_words, fake_words)
count_words("syria", real_words, fake_words)
count_words("finance", real_words, fake_words)
count_words("war", real_words, fake_words)
count_words("polls", real_words, fake_words)


# In[ ]:



