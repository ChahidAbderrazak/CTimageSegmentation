# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 19:45:35 2021

@author: Abderrazak Chahid
"""

import sys
import os
import numpy as np
import SimpleITK as sitk
import pickle
# from myLib import *
import nibabel as nib



class CT_data:
    
    def __init__(self, path_CT_dataset='../data/CT_dataset.npy', slice_size=[512,512] ):
        self.data_paths_dic, self.train_img_paths, self.train_label_paths, self.val_img_paths, self.val_label_paths, self.test_img_paths, self.test_label_paths=self.load_paths(path_CT_dataset)
        self.train_img, self.train_label, self.val_img, self.val_label, self.test_img, self.test_label=[],[],[],[],[],[]#np.zeros(slice_size[0],slice_size[1],1)

        
        
    def load_paths(self, path_CT_dataset):
        with open(path_CT_dataset, 'rb') as f:
            paths = pickle.load(f)
 
        return paths, paths.get('train_img'),paths.get('train_label'), paths.get('val_img'),paths.get('val_label'), paths.get('test_img'),paths.get('test_label')
    
    
    def load_data(self):
        
        cnt=0

        for set in [self.train_img_paths,self.val_img_paths, self.test_img_paths]:
            
            cnt=cnt+1
            
            for file in set:
                ID_pos=charposition(file,'_')
                # print(file)
                # print(ID_pos)
                
                patient=file[ID_pos[-1]+1:-3] 
                # print(patient)
                
                try:
                    ## load the stored data
                    volume=np.asanyarray(nib.load(file).dataobj)
                except: 
                    pass
                
                print('  --. processsing : ',file)

                # print('image', volume.shape)
                file_mask=file.replace('image', 'label')
                # print('mask?',os.path.isfile(file_mask))
                    
                if cnt==1:
                    
                    
                    
                    if os.path.isfile(file_mask):
                        
                        volume_mask=np.asanyarray(nib.load(file_mask).dataobj)
                        
                        # print(' The size of img=',volume.shape)
                        # print(' The size of mask=',volume_mask.shape)
                          
                        if volume_mask.shape==volume.shape:
                            
                            # print('exist')
                        
                            # training_image
                            if self.train_img==[]:
                                self.train_img=volume
                                self.train_label=volume_mask
                            else:
                                    
                                self.train_img=np.concatenate((self.train_img,volume),axis=2)
                                self.train_label=np.concatenate((self.train_label,volume_mask),axis=2)
                        else:
                            
                          print ("The sizes of image and masks are not matching in the file "+file )
                          

                            
                    else:
                        print ("The matching mask of "+file + " does not exixst!. Skipped data")

                    # print('new size=',self.train_img.shape)
    
    
    
    
                    
                elif cnt==2:

                    # validation_image
                    
                    file_mask=file.replace('image', 'label')
                    
                    if os.path.isfile(file_mask):
                        
                        volume_mask=np.asanyarray(nib.load(file_mask).dataobj)

                        # training_image
                        if self.val_img==[]:
                            self.val_img=volume
                            self.val_label=volume_mask
                        else:
                                
                            self.val_img=np.concatenate((self.val_img,volume),axis=2)
                            self.val_label=np.concatenate((self.val_label,volume_mask),axis=2)
                        
                    else:
                        print ("The matching mask of "+file + " does not exixst!. Skipped data")
                        
                        
                        
                        
                    
                elif cnt==3:
                    # testing_image
                    file_mask=file.replace('image', 'label')
                    
                # if os.path.isfile(file_mask):
                    
                #     volume_mask=np.asanyarray(nib.load(file_mask).dataobj)

                    # training_image
                    if self.test_img==[]:
                        self.test_img=volume
                        self.test_label=volume_mask
                    else:
                            
                        self.test_img=np.concatenate((self.test_img,volume),axis=2)
                    
                # else:
                #     print ("The matching mask of "+file + " does not exixst!. Skipped]
                           
                               
                               

        print('  --> Loading data is done')
        
        
    def print_paths(self):
        
         print(self.data_paths_dic)   
         
         print('training images=',self.train_label_paths) 
 
        
 
##/////////////////////////////////////////////////////////////////////////////////////////


# print('#############################################')
# print('##             Data Handler                ##')
# print('#############################################')

# ## dataset path file
# path_CT_dataset='../data/CT_dataset.np'

# # Data handler, load the data
# mydata=CT_data(path_CT_dataset)
# mydata.load_data()

# #mydata.print_paths()

