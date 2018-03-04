import torch
import torchvision 
import torchvision.models as models
import os

from torch.autograd import Variable

import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread, imresize

import torch.nn as nn


from caffe_classes import class_names

class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            
        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self.load_weights()
    
    '''Modified forward function. Returns actications of chosen layer.'''
    def forward(self, x):
        set = self.modifyNet(10)
        x = set(x)
        x = x.view(x.size(0), 256 *13 *13)
        return x
   
   
   '''Function to edit the features list to be able to extract a chosen layer's activations.Input is an index corresponding to the selected layer in the self.features list. Returns a modified self.features funtion'''
    def modifyNet(self,i):
        new_features = []
        for i in range(0,i+1):
            new_features.append(self.features[i])
            new = nn.Sequential(*new_features)
    return new




'''Get a training set. Returns two arrays. One with the flattened input images and the other with the corresponding one-hot encoding outputs for each actor'''
def get_train():
    batch_xs = 0
    batch_xs = []
    batch_y_s = np.zeros( (0, 6))
    
    
    
    actors = ['gilpin','bracco','baldwin', 'carell','hader', 'harmon']
    
    for k in range(0,6):
        count = 0
        for filename in os.listdir("./edit3"):
            if actors[k] in filename:
                 
                if (count < 50):
                    image = imread("./edit3/"+filename)
                    if image.shape == (32,32):
                        im = imresize(image, 7.10)
                        #im = np.hstack((im,im,im))
                        im = np.resize(im, (227, 227,3))
                        #print im.shape
                    else:
                        im = imresize(image,7.10)
                    im = im[:,:,:3]
                    im = im - np.mean(im.flatten())
                    im = im/np.max(np.abs(im.flatten()))
                    im = np.rollaxis(im, -1).astype(np.float32)
                    
                    batch_xs.append(im)
                    
                    one_hot = np.zeros(6)
                    one_hot[k] = 1
                    batch_y_s = np.vstack((batch_y_s, one_hot ))
                count +=1
    
    return batch_xs, batch_y_s


'''Get a test set. Returns two arrays. One with the flattened input images and the other with the corresponding one-hot encoding outputs for each actor'''
def get_test():
    batch_xs = 0
    batch_xs = []
    batch_y_s = np.zeros( (0, 6))
    
    
    
    actors = ['gilpin','bracco','baldwin', 'carell','hader', 'harmon']
    
    for k in range(0,6):
        count = 0
        for filename in os.listdir("./edit3"):
            if actors[k] in filename:
                 
                if (60 < count and count <= 80):
                    image = imread("./edit3/"+filename)
                    if image.shape == (32,32):
                        im = imresize(image, 7.10)
                        #im = np.hstack((im,im,im))
                        im = np.resize(im, (227, 227,3))
                        #print im.shape
                    else:
                        im = imresize(image,7.10)
                    im = im[:,:,:3]
                    im = im - np.mean(im.flatten())
                    im = im/np.max(np.abs(im.flatten()))
                    im = np.rollaxis(im, -1).astype(np.float32)
                    
                    batch_xs.append(im)
                    
                    one_hot = np.zeros(6)
                    one_hot[k] = 1
                    batch_y_s = np.vstack((batch_y_s, one_hot ))
                count +=1
    
    return batch_xs, batch_y_s


'''Get a validation set. Returns two arrays. One with the flattened input images and the other with the corresponding one-hot encoding outputs for each actor'''
def get_valid():
    batch_xs = 0
    batch_xs = []
    batch_y_s = np.zeros( (0, 6))
    
    
    
    actors = ['gilpin','bracco','baldwin', 'carell','hader', 'harmon']
    
    for k in range(0,6):
        count = 0
        for filename in os.listdir("./edit3"):
            if actors[k] in filename:
                 
                if (80 < count and count <= 100):
                    image = imread("./edit3/"+filename)
                    if image.shape == (32,32):
                        im = imresize(image, 7.10)
                        #im = np.hstack((im,im,im))
                        im = np.resize(im, (227, 227,3))
                        #print im.shape
                    else:
                        im = imresize(image,7.10)
                    im = im[:,:,:3]
                    im = im - np.mean(im.flatten())
                    im = im/np.max(np.abs(im.flatten()))
                    im = np.rollaxis(im, -1).astype(np.float32)
                    
                    batch_xs.append(im)
                    
                    one_hot = np.zeros(6)
                    one_hot[k] = 1
                    batch_y_s = np.vstack((batch_y_s, one_hot ))
                count +=1
    
    return batch_xs, batch_y_s


dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

model = MyAlexNet()     #First model to extract layers


'''New learning model that takes in activations as input'''
model2 = torch.nn.Sequential(
    nn.Linear(256 *13 * 13, 20),
    nn.Hardtanh(inplace=True),
    torch.nn.Linear(20,6)
)


loss_fn = torch.nn.CrossEntropyLoss()

xtest, ytest = get_test()   #Get test set
xvalid, yvalid = get_valid() #Get validation set

xtest = np.asarray(xtest)
xvalid = np.asarray(xvalid)

xaxis = [0,100,200,300,400,500,600,700,800]
train = [0]
test  = [0]
valid = [0]


im_v_test = Variable(torch.from_numpy(xtest), requires_grad=False)   #Convert sets into variables
im_v_valid = Variable(torch.from_numpy(xvalid), requires_grad=False)

in_test = (model.forward(im_v_test)).data.numpy()
in_valid = (model.forward(im_v_valid)).data.numpy()

in_test = Variable(torch.from_numpy(in_test), requires_grad = False).type(dtype_float)
in_valid = Variable(torch.from_numpy(in_valid), requires_grad = False).type(dtype_float)

model2[0].weight.data.fill_(0)


xtrain,ytrain = get_train()
xtrain = np.asarray(xtrain)


im_v = Variable(torch.from_numpy(xtrain), requires_grad=False)


softmax = torch.nn.Softmax()
in_put = (model.forward(im_v)).data.numpy()

in_put = Variable(torch.from_numpy(in_put), requires_grad = False).type(dtype_float)


y_classes = Variable(torch.from_numpy(np.argmax(ytrain,1)), requires_grad=False).type(dtype_long)



learning_rate = 1e-6
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)
for i in range(100,900,100):

    for t in range(i):
        y_pred = model2(in_put)
        loss = loss_fn(y_pred, y_classes)
    
        optimizer.zero_grad() #model.zero_grad()  # Zero out the previous gradient computation
        loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to 
                       # make a step

        

    model2.train()
    x = in_put
    y_pred = model2(x).data.numpy()

    res1 = np.mean(np.argmax(y_pred, 1) == np.argmax(ytrain, 1))
    print "Training Set:",res1
    train.append(res1)   #Add to training performance accuracy

    model2.eval()	
    x = in_test
    y_pred = model2(x).data.numpy()


    res2 = np.mean(np.argmax(y_pred, 1) == np.argmax(ytest, 1))
    print "Test set:",res2
    test.append(res2)   #Add to test set performance accuracy


    model2.eval()
    x = in_valid
    y_pred = model2(x).data.numpy()


    res3 = np.mean(np.argmax(y_pred, 1) == np.argmax(yvalid, 1))
    print "Validation Set:",res3
    valid.append(res3)    #Add to validation set performance accuracy


'''Create plot'''
plt.plot(xaxis,train,'red', label = 'Training Set')
plt.plot(xaxis,test, 'blue', label = 'Test Set')
plt.plot(xaxis,valid, 'green', label = 'Validation Set')


plt.legend(loc = 'lower right')
 
plt.xlabel("Iterations")
plt.ylabel("Performance Accuracy")
plt.show()



