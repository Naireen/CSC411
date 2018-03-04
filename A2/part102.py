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

    def forward(self, x):
        x = self.features(x)
        #print x
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def get_train():
    batch_xs = 0
    batch_xs = []
    batch_y_s = np.zeros( (0, 6))
    
    
    
    actors = ['gilpin'] #,'bracco','baldwin', 'carell','hader', 'harmon']
    
    for k in range(0,1):
        count = 0
        for filename in os.listdir("./edit3"):
            if actors[k] in filename:
                 
                if (count < 1):
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

xtrain, ytrain = get_train()
xtrain = np.asarray(xtrain)
#ytrain = np.zeros(6)
#ytrain[4] =1 
#print (xtrain[5].shape)

# Read an image
im = imread('kiwi227.png')[:,:,:3]
im = im - np.mean(im.flatten())
im = im/np.max(np.abs(im.flatten()))
im = np.rollaxis(im, -1).astype(np.float32)

# turn the image into a numpy variable
im_v = Variable(torch.from_numpy(xtrain), requires_grad=False)
model = MyAlexNet()


print im_v.shape

softmax = torch.nn.Softmax()
model.forward(im_v)


print (model.features[10].weight).shape

xin = ((model.features[10].weight))
xin = xin.view(1,256*256*9)


print 'ok'


model2 = torch.nn.Sequential(
    nn.Linear(256 *256 * 9, 100),
    nn.ReLU(inplace=True),
    torch.nn.Linear(100,6),
)

loss_fn = torch.nn.CrossEntropyLoss()



y_classes = Variable(torch.from_numpy(np.argmax(ytrain,1)), requires_grad=False).type(dtype_long)
print y_classes.shape

learning_rate = 1e-6
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)

for t in range(10):
        print 'hey'
	y_pred = model2(xin)
        print y_pred
	loss = loss_fn(y_pred, y_classes)
    
    	optimizer.zero_grad() #model.zero_grad()  # Zero out the previous gradient computation
    	loss.backward()    # Compute the gradient
        optimizer.step()   # Use the gradient information to 
                       # make a step

        print "hello"

model2.train()
x = im_v
y_pred = model(x).data.numpy()



res1 = np.mean(np.argmax(y_pred, 1) == np.argmax(ytrain, 1))
print res1

