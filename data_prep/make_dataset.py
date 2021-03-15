# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:55:52 2021

@author: Abderrazak Chahid
"""


import sys
import os
import numpy as np
import SimpleITK as sitk
import gzip
import shutil
from sklearn.model_selection import train_test_split
from data_prep import *
import pickle

## Routines

def charposition(string, char):
    pos = [] #list to store positions for each 'char' in 'string'
    for n in range(len(string)):
        if string[n] == char:
            pos.append(n)
    return pos


def extract_save_CT_data(root_folder):
    
    
    from glob import glob
    folders= glob(root_folder+"/*/")
    
    print(folders)

    file_type = ".gz"

    
    for data_folder in folders:
        data_folder=data_folder.replace("\\",'/')
        dst_folder=data_folder
        
        print('prparing the dataset from the folder:',data_folder)
    
    
        ## building the dataset
        list_img, list_label, list_patients= [],[],[]
    
        if data_folder[-10:].find('train') !=-1:
            flag_train=1
            
        else:
            flag_train=0
        
        
        
        cnt=1
        for fname in os.listdir(path=data_folder):
            if fname.endswith(file_type):
                gz_file=data_folder+fname
                
                # gte the patient name
                ID_pos=charposition(gz_file,'_')
                patient='p'+gz_file[ID_pos[-2]+1:ID_pos[-1]]
                list_patients.append(patient)
    
                print('\n\n Processing: '+ patient+ '  file:  ' +gz_file)
                
                
                ## ectract the nii format form gz
                with gzip.open(gz_file,'rb') as f_in:
                    
                    ## extract the nii from gz in 
                    nii_file=dst_folder+fname[:-3]
                    
                    # save the nii format 
                    with open(nii_file,'wb') as f_out:
                        shutil.copyfileobj(f_in,f_out)
                        
                        
                    ## save tyhe np format 
                    load_nii_save_np_data(nii_file, data_folder)
                    


def make_CT_data(root_folder):
    
    from glob import glob
    folders= glob(root_folder+"/*/")
    # print(folders)
    
    ## building the dataset
    list_img_train, list_label_train, list_img_test, list_label_test, list_patients= [],[],[],[],[]    
    
    
    for data_folder in folders:    
        data_folder=data_folder.replace("\\",'/')
        dst_folder=data_folder
    
  
    
        cnt=1
        for fname in os.listdir(path=data_folder):
            if fname.endswith(".nii"):
                np_file=data_folder+fname
                
                # gte the patient name
                ID_pos=charposition(np_file,'_')
                # print(np_file)
                # print(ID_pos)
                
                patient=np_file[ID_pos[-1]+1:-3]
                list_patients.append(patient)
                # print(list_patients)

                if data_folder[-10:].find('train') !=-1:
            
                    ## file the files paths in the dataset
                    if np_file.find('image') !=-1:
                        list_img_train.append(np_file); 
                        print('Adding training Image of patient : '+ patient+ '  from file:  ' +np_file)
                        
                        
                    if np_file.find('label') !=-1:
                       list_label_train.append(np_file)
                       print('Adding training Mask of patient : '+ patient+ '  from file:  ' +np_file)
                        
                if data_folder[-10:].find('test') !=-1:
            
                    ## file the files paths in the dataset
                    if np_file.find('image') !=-1:
                        list_img_test.append(np_file); 
                        print('Adding testing Image of patient : '+ patient+ '  from file:  ' +np_file)
                        
                    if np_file.find('label') !=-1:
                       list_label_test.append(np_file)
                       print('Adding testing Mask of patient : '+ patient+ '  from file:  ' +np_file)
     
    return list_img_train, list_label_train, list_img_test, list_label_test, list_patients
                
def save_dict(di_, filename_):
   with open(filename_, 'wb') as f:
       pickle.dump(di_, f)

#%%/////////////////////////////////////////////////////////////////////////////////////////
   
print('#############################################')
print('##                        Making  Dataset                 ##')
print('#############################################')

## Get the  full paths of the  training np data 
data_folder ='../../data/'
print('\n CT-dataset source is :', data_folder)


## conversion 
print('--. Convert the .gz to nii ')
extract_save_CT_data(data_folder)

#%% saving dataset paths
training_img, training_label, test_img, test_label, training_patients=make_CT_data(data_folder)

## random splitting the training data
train_img, val_img, train_label, val_label = train_test_split(training_img, training_label, test_size = 0.2) #splitting the dataset

## build and save dataset paths
CT_dataset={'train_img':train_img, 'train_label':train_label,
            'val_img':val_img, 'val_label':val_label,
            'test_img':test_img, 'test_label':test_label}

save_dict(CT_dataset, data_folder+'CT_dataset.np')

print('\n CT_dataset paths are rendomly resorted and saved:')
print('   Training='+str(len(train_img))+', Validation='+str(len(val_img))+', Testing='+str(len(test_img)))