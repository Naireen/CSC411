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

'''
def get_test():
    batch_xs = np.zeros((0, 32*32))
    batch_y_s = np.zeros( (0, 6))
    
    actors = ['gilpin','bracco','baldwin', 'carell','hader', 'harmon']
    
    for k in range(0,6):
        count = 0
        for filename in os.listdir("./edit3"):
            if actors[k] in filename:
                count +=1 
                if (60 <count and count <= 80):
                    image = imread("./edit2/"+filename)
                    x = image.flatten()/255.0
                    batch_xs = np.vstack((batch_xs, x))
                    
                    one_hot = np.zeros(6)
                    one_hot[k] = 1
                    batch_y_s = np.vstack((batch_y_s, one_hot ))
    return batch_xs, batch_y_s'''

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
                        
                        #image = image.append(image,image)		
			#print x.shape

                    elif image.shape[2] == 4:
			image = image[:,:,0:3]
                        x = image.flatten()/255.0
                    else:
                        x = image.flatten()/255.0

                    #print x.shape
		    batch_xs = np.vstack((batch_xs, x))
                    
                    one_hot = np.zeros(6)
                    one_hot[k] = 1
                    batch_y_s = np.vstack((batch_y_s, one_hot ))
                count +=1
    return batch_xs, batch_y_s
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
                        
                        #image = image.append(image,image)		
			#print x.shape

                    elif image.shape[2] == 4:
			image = image[:,:,0:3]
                        x = image.flatten()/255.0
                    else:
                        x = image.flatten()/255.0

                    #print x.shape
		    batch_xs = np.vstack((batch_xs, x))
                    
                    one_hot = np.zeros(6)
                    one_hot[k] = 1
                    batch_y_s = np.vstack((batch_y_s, one_hot ))
                count +=1
    return batch_xs, batch_y_s

def get_train(i):
    batch_xs = np.zeros((0, 1024*3))
    batch_y_s = np.zeros( (0, 6))
    
    
    
    actors = ['gilpin','bracco','baldwin', 'carell','hader', 'harmon']
    
    for k in range(0,6):
        count = 0
        for filename in os.listdir("./edit3"):
            if actors[k] in filename:
                 
                if (count < i):
                    image = imread("./edit3/"+filename)
                    if image.shape == (32,32):
                        x = image.flatten()/255.0
                        x = np.hstack((x,x,x))
                        
                        #image = image.append(image,image)		
			#print x.shape

                    elif image.shape[2] == 4:
			image = image[:,:,0:3]
                        x = image.flatten()/255.0
                    else:
                        x = image.flatten()/255.0

                    #print x.shape
		    batch_xs = np.vstack((batch_xs, x))
                    
                    one_hot = np.zeros(6)
                    one_hot[k] = 1
                    batch_y_s = np.vstack((batch_y_s, one_hot ))
                count +=1
    return batch_xs, batch_y_s


xtest, ytest = get_test()
xvalid, yvalid = get_valid()
#print xtrain.shape
#print ytrain.shape
dim_x = 1024*3
print dim_x
dim_h = 12
dim_out = 6

dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor

xaxis = [0,10,20,30,40,50,60]
train = [0]
test  = [0]
valid = [0]

################################################################################
#Subsample the training set for faster training
'''
train_idx = np.random.permutation(range(xtrain.shape[0]))
x = Variable(torch.from_numpy(xtrain[train_idx]), requires_grad=False).type(dtype_float)
print train_idx #
print x.shape   #
y_classes = Variable(torch.from_numpy(np.argmax(ytrain[train_idx], 1)), requires_grad=False).type(dtype_long)'''
#################################################################################



model = torch.nn.Sequential(
    torch.nn.Linear(dim_x, dim_h),
    torch.nn.Hardtanh(),
    torch.nn.Linear(dim_h, dim_out),
)

loss_fn = torch.nn.CrossEntropyLoss()


for i in range(10,70,10):
        xtrain,ytrain = get_train(i)
        
	train_idx = np.random.permutation(range(xtrain.shape[0]))
	x = Variable(torch.from_numpy(xtrain[train_idx]), requires_grad=False).type(dtype_float)
	#print train_idx #
	print x.shape   #
	y_classes = Variable(torch.from_numpy(np.argmax(ytrain[train_idx], 1)), requires_grad=False).type(dtype_long)
	learning_rate = 1e-6
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	for t in range(10000):
	    y_pred = model(x)
	    loss = loss_fn(y_pred, y_classes)
    
    	    optimizer.zero_grad() #model.zero_grad()  # Zero out the previous gradient computation
    	    loss.backward()    # Compute the gradient
            optimizer.step()   # Use the gradient information to 
                       # make a step



	model.train()
    	x = Variable(torch.from_numpy(xtrain), requires_grad=False).type(dtype_float)
        y_pred = model(x).data.numpy()



        res1 = np.mean(np.argmax(y_pred, 1) == np.argmax(ytrain, 1))
        print res1
        train.append(res1)

        model.eval()
        x = Variable(torch.from_numpy(xtest), requires_grad=False).type(dtype_float)
        y_pred = model(x).data.numpy()



        res = np.mean(np.argmax(y_pred, 1) == np.argmax(ytest, 1))
        test.append(res)

        print res
        
        x = Variable(torch.from_numpy(xvalid), requires_grad=False).type(dtype_float)
        y_pred = model(x).data.numpy()



        res3 = np.mean(np.argmax(y_pred, 1) == np.argmax(yvalid, 1))
        

        print res3
	
	valid.append(res3)

plt.plot(xaxis,train)
plt.plot(xaxis,test)
plt.plot(xaxis,valid)
#print model[0].weight
#model[0].weight.data.numpy()[10, :].shape

#plt.imshow(model[0].weight.data.numpy()[4, :].reshape((32, 32)), cmap=plt.cm.coolwarm)
plt.show()

