from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt
import os


from scipy.io import loadmat

from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
from scipy.misc import imresize

# %matplotlib inline  

'''Get a validation set. Returns two arrays. One with the flattened input images and the other with the corresponding one-hot encoding outputs for each actor'''
def get_valid():
    batch_xs = np.zeros((0, 1024*3))
    batch_y_s = np.zeros( (0, 6))
    
    
    
    actors = ['gilpin','bracco','baldwin', 'carell','hader', 'harmon']
    
    for k in range(0,6):
        count = 0
        for filename in os.listdir("./edit3"):
            if actors[k] in filename:
                 
                if (80< count and count <= 100):
                    image = imread("./edit3/"+filename)
                    if image.shape == (32,32):
                        x = image.flatten()/255.0
                        x = np.hstack((x,x,x))
                        
            

                    elif image.shape[2] == 4:
                        image = image[:,:,0:3]
                        x = image.flatten()/255.0
                    else:
                        x = image.flatten()/255.0

                    
                    batch_xs = np.vstack((batch_xs, x))
                    
                    one_hot = np.zeros(6)
                    one_hot[k] = 1
                    batch_y_s = np.vstack((batch_y_s, one_hot ))
                count +=1
    return batch_xs, batch_y_s


'''Get a test set. Returns two arrays. One with the flattened input images and the other with the corresponding one-hot encoding outputs for each actor'''
def get_test():
    batch_xs = np.zeros((0, 1024*3))
    batch_y_s = np.zeros( (0, 6))
    
    
    
    actors = ['gilpin','bracco','baldwin', 'carell','hader', 'harmon']
    
    for k in range(0,6):
        count = 0
        for filename in os.listdir("./edit3"):
            if actors[k] in filename:
                 
                if (60< count and count <= 80):
                    image = imread("./edit3/"+filename)
                    if image.shape == (32,32):
                        x = image.flatten()/255.0
                        x = np.hstack((x,x,x))
                        
              

                    elif image.shape[2] == 4:
                        image = image[:,:,0:3]
                        x = image.flatten()/255.0
                    else:
                        x = image.flatten()/255.0

                    
                    batch_xs = np.vstack((batch_xs, x))
                    
                    one_hot = np.zeros(6)
                    one_hot[k] = 1
                    batch_y_s = np.vstack((batch_y_s, one_hot ))
                count +=1
    return batch_xs, batch_y_s


'''Get a training set. Returns two arrays. One with the flattened input images and the other with the corresponding one-hot encoding outputs for each actor'''
def get_train():
    batch_xs = np.zeros((0, 1024*3))
    batch_y_s = np.zeros( (0, 6))
    
    
    
    actors = ['gilpin','bracco','baldwin', 'carell','hader', 'harmon']
    
    for k in range(0,6):
        count = 0
        for filename in os.listdir("./edit3"):
            if actors[k] in filename:
                 
                if (count < 50):
                    image = imread("./edit3/"+filename)
                    if image.shape == (32,32):
                        x = image.flatten()/255.0
                        x = np.hstack((x,x,x))
                        
                       

                    elif image.shape[2] == 4:
                        image = image[:,:,0:3]
                        x = image.flatten()/255.0
                    else:
                        x = image.flatten()/255.0

                    
                        batch_xs = np.vstack((batch_xs, x))
                    
                    one_hot = np.zeros(6)
                    one_hot[k] = 1
                    batch_y_s = np.vstack((batch_y_s, one_hot ))
                count +=1
    return batch_xs, batch_y_s


xtest, ytest = get_test()
xvalid, yvalid = get_valid()


dim_x = 1024*3
dim_h = 12
dim_out = 6

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

xaxis = [0,10,20,30,40,50,60]


model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.ReLU(),
    torch.nn.Linear(dim_h, dim_out),
)


loss_fn = torch.nn.CrossEntropyLoss()
xtrain,ytrain = get_train()  #get training set
        
train_idx = np.random.permutation(range(xtrain.shape[0]))
x = Variable(torch.from_numpy(xtrain[train_idx]), requires_grad=False).type(dtype_float)
y_classes = Variable(torch.from_numpy(np.argmax(ytrain[train_idx], 1)), requires_grad=False).type(dtype_long)

learning_rate = 1e-6
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(1000):
    y_pred = model(x)
    loss = loss_fn(y_pred, y_classes)
    
    optimizer.zero_grad() #model.zero_grad()  # Zero out the previous gradient computation
    loss.backward()    # Compute the gradient
    optimizer.step()   # Use the gradient information to 
                       # make a step


#visualize weights
plt.subplot(4,6,1)

image1 = (model[0].weight.data.numpy()[0,:1024]).reshape((32,32))
#image2 = (model[0].weight.data.numpy()[0,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[0,2048:3072]).reshape((32,32))
#image = [image1,image2,image3]
plt.imshow((image1), cmap=plt.cm.coolwarm)

plt.subplot(4,6,2)
image1 = (model[0].weight.data.numpy()[1,0:1024]).reshape((32,32))
image2 = (model[0].weight.data.numpy()[1,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[23,2048:3072]).reshape((32,32))
plt.imshow((image2), cmap=plt.cm.coolwarm)

plt.subplot(4,6,3)
image1 = (model[0].weight.data.numpy()[2,0:1024]).reshape((32,32))
#image2 = (model[0].weight.data.numpy()[23,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[23,2048:3072]).reshape((32,32))
plt.imshow((image1), cmap=plt.cm.coolwarm)

plt.subplot(4,6,4)
image1 = (model[0].weight.data.numpy()[3,0:1024]).reshape((32,32))
#image2 = (model[0].weight.data.numpy()[23,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[23,2048:3072]).reshape((32,32))
plt.imshow((image1), cmap=plt.cm.coolwarm)

plt.subplot(4,6,5)
image1 = (model[0].weight.data.numpy()[4,0:1024]).reshape((32,32))
#image2 = (model[0].weight.data.numpy()[23,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[23,2048:3072]).reshape((32,32))
plt.imshow((image1), cmap=plt.cm.coolwarm)

plt.subplot(4,6,6)
image1 = (model[0].weight.data.numpy()[5,0:1024]).reshape((32,32))
#image2 = (model[0].weight.data.numpy()[23,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[23,2048:3072]).reshape((32,32))
plt.imshow((image1), cmap=plt.cm.coolwarm)

plt.subplot(4,6,7)
image1 = (model[0].weight.data.numpy()[6,0:1024]).reshape((32,32))
#image2 = (model[0].weight.data.numpy()[23,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[23,2048:3072]).reshape((32,32))
plt.imshow((image1), cmap=plt.cm.coolwarm)

plt.subplot(4,6,8)
image1 = (model[0].weight.data.numpy()[7,0:1024]).reshape((32,32))
#image2 = (model[0].weight.data.numpy()[23,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[23,2048:3072]).reshape((32,32))
plt.imshow((image1), cmap=plt.cm.coolwarm)

plt.subplot(4,6,9)
image1 = (model[0].weight.data.numpy()[8,0:1024]).reshape((32,32))
#image2 = (model[0].weight.data.numpy()[23,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[23,2048:3072]).reshape((32,32))
plt.imshow((image1), cmap=plt.cm.coolwarm)

plt.subplot(4,6,10)
image1 = (model[0].weight.data.numpy()[9,0:1024]).reshape((32,32))
#image2 = (model[0].weight.data.numpy()[23,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[23,2048:3072]).reshape((32,32))
plt.imshow((image1), cmap=plt.cm.coolwarm)

plt.subplot(4,6,11)
image1 = (model[0].weight.data.numpy()[10,0:1024]).reshape((32,32))
#image2 = (model[0].weight.data.numpy()[23,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[23,2048:3072]).reshape((32,32))
plt.imshow((image1), cmap=plt.cm.coolwarm)

plt.subplot(4,6,12)
image1 = (model[0].weight.data.numpy()[11,0:1024]).reshape((32,32))
#image2 = (model[0].weight.data.numpy()[23,1024:2048]).reshape((32,32))
#image3 = (model[0].weight.data.numpy()[23,2048:3072]).reshape((32,32))
plt.imshow((image1), cmap=plt.cm.coolwarm)


plt.show()

