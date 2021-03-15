# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:53:41 2021

@author: Abderrazak Chahid
"""


import sys
import os
import numpy as np
import pickle
import SimpleITK as sitk

import gzip
import shutil

def save_np_var(var_, filename_):
    
   with open(filename_, 'wb') as f:
       pickle.dump(var_, f)

    

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
                    

def load_nii_save_np_data(filename, folder_dst):
    ID_pos=charposition(filename,'_')
    # print(filename)
    # print(ID_pos)
    patient='p'+filename[ID_pos[-2]+1:ID_pos[-1]]
    print("Reading DICOM data of the patient:"+ patient+'   please wait....')
    
    # setup SimpleITK reader
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    
    # get the Heater in np dicrionary
    hd_ct = []
        
    for k in reader.GetMetaDataKeys():
        v = reader.GetMetaData(k)
        meta = {k: v }
        hd_ct.append(meta)
            
    hd= np.array(hd_ct)
    
     # Read image values in  .nii image 
    img_ct = sitk.ReadImage(filename)
    img = sitk.GetArrayFromImage(img_ct)
        
    if filename.find('image') !=-1:

        ## save img and header
        save_np_var(img, folder_dst+'/img_'+patient+'.np')
        save_np_var(hd, folder_dst+'/hd_'+patient+'.np')
        
        # np.save(folder_dst+'/img_'+patient+'.np',img)
        # np.save(folder_dst+'/hd_'+patient+'.npy',hd)
    
        print("DICOM Image writing.")
        print('Numpy images and their headers are saved in folder:',folder_dst)
        
        
    else:

        ## save label "masks" 
        
        save_np_var(img, folder_dst+'/mask_'+patient+'.np')
        save_np_var(hd, folder_dst+'/mask_hdr_'+patient+'.np')
        
        # np.save(folder_dst+'/mask_'+patient+'.npy',img)
        # np.save(folder_dst+'/mask_hdr_'+patient+'.npy',hd)
        print("DICOM Label writing:")
        print('Numpy masks and their headers are saved in folder:',folder_dst)
        
   # print("Header:",hd)
   # print("DICOM Image size:",img.shape)
    
    
 ##/////////////////////////////////////////////////////////////////////////////////////////

   
# print('#############################################')
# print('##            preparing np Dataset         ##')
# print('#############################################')
# ## Get thes training data ull paths
# data_folder ='../data'

# extract_save_CT_data(data_folder)


# ## input variables
# filename='../data/ct_test/ct_test_2001_image.nii'
# folder_dst='../data'


# ## numpy data preparation
# patient='p001'
# load_nii_save_np_data(filename, folder_dst)

# # print("Header:",hd)
# # print("DICOM Image size:",img.shape)

# print('#######  THE END #########')