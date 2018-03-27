#!/usr/bin/python3
#Note, must be in same directory as the saved model weights and dictionary  attached on markus
#import statements
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import pickle
from sklearn import metrics
import spacy
import sys
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)



data_path = sys.argv[1]
#print(data_path)

def make_data(data_path):
    good = open(data_path, "r")
    words = good.read()
    words = words.split("\n")
    #print(len(words))
    #total = len(words)
    #np.random.seed(0)
    #np.random.shuffle(words)
    return words


# 1 Number of first-person pronouns
# 2 Number of second-person pronouns
# 3 Number of third-person pronouns
# 4 Number of coordinating conjunctions
# 5 Number of past-tense verbs
# 6 Number of future-tense verbs
# 7 Number of common nouns
# 8 Number of proper nouns
# 9 Number of adverbs
# 10 who words
# 11 Average length of sentences, in tokens

nlp = spacy.load("en")

def make_feats(lines):
    data =np.zeros((len(lines), 11))
    first_person = ["I", 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    second_person = ['you', 'your', 'yours', 'u', 'ur', 'urs']
    third_person =['he', 'min', 'his','she', 'her', 'hers', 'it', 'its', 'they' , 'them' ,'their', 'theirs']
    future_tense = ["'ll", 'will', 'gonna','going to']
    for i, line in enumerate(lines):
        doc = spacy.tokens.Doc(nlp.vocab, words = line)
        tokens = nlp(line)
        total = 0
        for token in tokens:
            total+=1
            if token.text in first_person:
                data[i, 0] +=1
            elif token.text in second_person:
                data[i, 1] +=1
            elif token.text in third_person:
                data[i, 2] +=1
            elif token.text in future_tense:
                data[i, 5] +=1
            if token.tag_ == "CC":
                data[i, 3] +=1
            elif token.tag_ == "VBD":
                data[i, 4] +=1
            elif token.tag_ == "NNS" or token.tag_== "NN":
                data[i, 6] +=1
            elif token.tag_ == "NNPS" or token.tag_ == "NNP" :
                data[i, 7]
            elif token.tag_ == "RB" or token.tag_ == "RBR" or token.tag_ == "RBS":
                data[i, 8]+=1
            elif token.tag_ == "WDT" or token.tag_ =="WP" or token.tag_ == "WP$" or token.tag_=="WRB":
                data[i, 9] +=1
        data[i, 10] = total
    return data


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.features = nn.Sequential(
        # an affine operation: y = Wx + b
        nn.Linear(4818+11, 200),
        nn.ReLU(),
        #nn.Linear(2000, 200),
        #nn.ReLU(),
        #nn.Dropout( 0.1),
        nn.Linear(200, 20),
        nn.ReLU(),
        nn.Linear(20, 2), # final two classees, real or fake
    )
    def forward(self, x):
        # If the size is a square you can only specify a single number
        x = self.features(x)
        return x

def load_weights(policy, load_path):
    weights = torch.load(load_path)
    policy.load_state_dict(weights)

# Set up neural net
net = Net()
load_path = "bonus_model_weights.pkl"
load_weights(net, load_path)
#print(net)

# Create data_list from file path
#print("Load Data")
data = make_data(data_path)

# Create original features
#print("Create Features")
unique_keys = pickle.load(open("train_set_keys.pkl", "rb"))
valid_feats = np.zeros((len(data), len(unique_keys)))
for i , line in enumerate(data):
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



# Create Spacy features
#print("Create Spacy Features")
spacy_data = make_feats(data)

# Combine the two
new_test_X = np.concatenate((valid_feats, spacy_data), axis = 1)
new_test_X = Variable(torch.from_numpy(new_test_X), requires_grad = False).type(torch.FloatTensor)
# Create predictions based on model and data
#print("Predictions")
preds = net(new_test_X)
preds = torch.max(preds,1)
preds = preds[1].data # this is a n by 1 array of data
#print(preds)
for value in preds:
    print(value)










