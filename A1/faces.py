#Assume that you give the TA the uncropped and cropped  folders to work with
#Need to produced the processed folder, and then the train, test and validation folder
#uncropped and cropped  are produced when downloading the 6 actors
#uncropped2 and  cropped 2 are created when downloading the other 6 actors not included in act
#run Recolor and Resize twice, with the appropriate args
#run sort data to create the file folders
#run both Create Data files.py to get the txt files needed 
#Run Gradient Descent
#Run Multiclassification

import subprocess
#import Recolor_Resize # finish processing after dowloads
import os
try:
    import sort_data #sort data
    print("Error free")
except Exception as e:
    print(e)    
print("Data should be sorted")
os.system("ls Train/ |grep -v Or* > Train/Order.txt " )
os.system("ls Test/ |grep -v Or*   > Test/Order.txt " )
os.system("ls Validation/ |grep -v Or*   > Validation/Order.txt " )


#import Create_Data_Files_6_More # create the 6 additional for testing
import Create_Data_Files # create the datafiles for the 6 actors
#import Create_Data_Files_6_More

#from subprocess import call as call

os.system("ls processed2/chenoweth* | xargs -n 1 basename > processed2/Order.txt")
os.system("ls processed2/ferrera* | xargs -n 1 basename > processed2/Order2.txt")
os.system("ls processed2/drescher* | xargs -n 1 basename > processed2/Order3.txt")
os.system("ls processed2/radcliffe* | xargs -n 1 basename > processed2/Order4.txt")
os.system("ls processed2/butler* | xargs -n 1 basename > processed2/Order5.txt")
os.system("ls processed2/vartan* | xargs -n 1 basename > processed2/Order6.txt")
import Create_Data_Files_6_More

# you can now run the gradient descent stuff







#import statements
#import os
#import matplotlib
#import time
#import numpy as np



