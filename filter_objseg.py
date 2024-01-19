# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:47:44 2024

@author: Boyang
"""

import trimesh
import os
import numpy as np
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

def show_distributon(data):
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    
    # Displaying statistics
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Standard Deviation: {std_dev}")
    
    # Creating a histogram
    plt.hist(data, bins=10, edgecolor='black')
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    # Show the histogram
    plt.show()


if __name__ == '__main__':

    source="./tee_300_raw"
    source_train=os.path.join(source, "train")
    
    target="./tee_300"
    target_train=os.path.join(target, "train")
        
    if not os.path.exists(target_train):
        os.makedirs(target_train)
    
    
    obj_files = [f[:-4] for f in os.listdir(source_train) if f.endswith('.obj')]
    
    lengthes=[]
    basenames=[]
    for file in tqdm(obj_files):
        mesh=trimesh.load(os.path.join(source_train,file+".obj"), process=False)
        lengthes.append(len(mesh.edges_unique))
        basenames.append(file)
    
    
    
    data=np.array(lengthes)
    
    #show_distributon(data) #which helps to get the lower_bound and higher_bound
    
    
    # filter the meshes of which the number of edges are inside the interval defined by 
    # lower_bound and higher_bound, and copy them to a new folder
    lower_bound=40000
    upper_bound=60000
    indices=np.where((data >= lower_bound) & (data <= upper_bound))[0]
    count = np.sum((data >= lower_bound) & (data <= upper_bound))
    print(count, len(indices))
    #print(indices[0])
    

    
    for idx in tqdm(indices):
        name=basenames[idx]    
        
        shutil.copyfile(os.path.join(source_train, name+".obj"), os.path.join(target_train,name+".obj"))
        shutil.copyfile(os.path.join(source,name+"_segmentation.txt"), os.path.join(target, name+"_segmentation.txt"))
    print("done")