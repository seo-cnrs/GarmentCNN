# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 17:46:57 2024

@author: Boyang YU
"""

import os  
import sys 
import shutil


def fullto3d(src,des):
    with open(src,'r') as f1, open(des,'a+') as f2:
            lines=f1.readlines()
            for line in lines:
                a=line.replace('\n','').split(" ")[:4]
                if a[0]=="v":
                    a[3]+="\n"
                    newline=" ".join(a) 
                    f2.write(newline)
                if a[0]=="f":
                    #print(a)
                    a[1]=a[1].split("/")[0]
                    a[2]=a[2].split("/")[0]
                    a[3]=a[3].split("/")[0]
                    newline=" ".join(a)
                    f2.write(newline+ '\n')


if __name__ == '__main__':
    
    path=sys.argv[1]
    target=sys.argv[2]
    nb_samples=sys.argv[3]
    
    print(path, target, nb_samples)
    
    """
    path="./tee_test"
    target= "./tee_samples"
    nb_samples=2 # the number of instances you want to extract
    """
    try:
        os.makedirs(target+"/train")
    except FileExistsError:
        # directory already exists
        pass
    
                    
    # copy and extract                 
    folders = [x[0] for x in os.walk(path)][1:]
    for i, folder in enumerate(folders) :
        if i == nb_samples:
            break
        print(folder)
        
        obj_files = [f for f in os.listdir(folder) if f.endswith('.obj')]
        text_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        for obj in obj_files:
            if("_scan_imitation" not in obj ):
                #shutil.copyfile(str(file)+'/'+obj, target+"/"+obj)
                #shutil.copyfile(str(file)+'/'+obj, target+'/train/'+obj)
                src= folder+'/'+obj
                des=target+'/train/'+obj
                fullto3d(src,des)
                
        for text_file in text_files:
            if("_scan_imitation" not in text_file ):
                shutil.copyfile(folder+'/'+text_file,target+"/"+text_file)
                
    print("done")