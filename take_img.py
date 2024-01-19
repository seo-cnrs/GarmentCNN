# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:30:12 2024

@author: Lenovo
"""
import shutil,os
import sys                

if __name__ == '__main__':
    # python take_img.py ./tee_2300 ./tee_100_imgs 100
    # python take_img.py ./tee_test ./tee_1_imgs 1
    
    path=sys.argv[1]
    target=sys.argv[2]
    nb_samples=sys.argv[3]
    
    if not os.path.exists(target):
        os.makedirs(target)
    
    print('take %s images from %s to %s for front view check' %
                      (nb_samples, path, target))
    

    
    # copy and extract                 
    folders = [x[0] for x in os.walk(path)][1:]
    for i, folder in enumerate(folders) :
        if i == int(nb_samples):
            break
        print(folder)
        
        image_files = [f for f in os.listdir(folder) if f[-9:-4]=="front"]
        #text_files = [f for f in os.listdir(folder) if f.endswith('.txt')]
        for image in image_files:
            print(image)
            shutil.copy(os.path.join(folder,image),os.path.join(target,image))
            
    print("done")
