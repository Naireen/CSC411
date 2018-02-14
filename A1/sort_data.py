import os
import matplotlib.image as mpimg
import numpy as np
from shutil import copyfile
from matplotlib import pyplot as plt


#act = list(set([a.split("\t")[0].replace("\n", "") for a in open("subset_actors.txt").readlines()]))

act = ["Alec Baldwin", "Lorraine Bracco" , "Steve Carell", "Perri Gilpin", "Bill Hader", "Angie Harmon"]


print(act)
files = os.listdir("processed/")

print("Processed file length", len(files))
for name in act:
    counter = 0
    print(name)
    for file_name in files:
        #print (name.split(" ")[1].lower())
        if name.split(" ")[1].lower() in file_name:
            #if the actor is in the file name of pic
            #sort the images in approriate spot accordingly
            if counter > 130:
                break
            try:
                #pic = mpimg.imread("processed/" + file_name)
                #print("Successfully loaded")
                if counter <110:
                    #mpimg.imsave("Train/" + file_name, pic, cmap = plt.cm.gray)
                    copyfile("processed/" + file_name, "Train/" + file_name)
                elif counter<120: 
                    #mpimg.imsave("Validation/" + file_name, pic, cmap = plt.cm.gray)
                    copyfile("processed/" + file_name, "Validation/" + file_name)
                elif counter<130:
                    #mpimg.imsave("Test/" + file_name, pic, cmap = plt.cm.gray)
                    copyfile("processed/" + file_name, "Test/" + file_name)
                counter +=1
                print(counter)
            except Exception as e:
                print(e)
            #once while loop concludeds,
            if counter > 130:
                break

