import sys
import os
import numpy as np
import pickle
from data_handler import *
import nibabel as nib

import gzip
import shutil

import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import model_from_json

from models import *
from data_prep.data_prep import *
#%%

def load_nii_data(filename):
    ID_pos=charposition(filename,'_')
    # print(filename)
    # print(ID_pos)
    patient='p'+filename[ID_pos[-2]+1:ID_pos[-1]]
    print("Reading DICOM data of the patient:"+ patient+'   please wait....')
    
    img=np.asanyarray(nib.load(filename).dataobj)
    
    return img

def save_nii_data(filename, var):
    
    img = nib.Nifti1Image(data, np.eye(4))
    img.get_data_dtype() == np.dtype(np.int16)
    img.header.get_xyzt_units()
    


    
    img=np.asanyarray(nib.load(filename).dataobj)
    
    return img


def plot_slice_test( img, predicted_mask):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
        
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
       
    ax1.imshow(img, cmap='gray')
    plt.title('Input image Mask')
    
    
    ax2 = fig.add_subplot(122) 
    ax2.imshow(predicted_mask)
    plt.title('Predicted Mask')
  
#%%      
print('#############################################')
print('##              Unet  Deployment           ##')
print('#############################################')



folder_nii='../data/ct_test/'
save_nii='../data/deploy/'
model_path='../data/models/Unet_weights_sizeH64_tag1.h5'
#%%
#%%
print('Dataset loaded ad reshaped [for computational reasons]')
x0=300; y0=300
imgWidth = 64*1
imgHeight = 64*1
num_classes=4
#%% Load the trained model 
# model_trained = Unet(imgWidth, imgHeight, 1, num_classes)# keras.Model(inputs=[inputs], outputs=[outputs])
model_trained = keras.models.load_model(model_path)

#%% 
for fname in os.listdir(path=folder_nii):
    
    if fname.endswith('.nii'):
        print('--> predicting the masks of the file:', fname)
        nii_file=folder_nii+fname
        
        print(nii_file)
                    
        # load the testing nii image
        img0=load_nii_data(nii_file)
        dataChannels=img0.shape[2]
        img=img0[x0:x0+imgWidth,y0:y0+imgHeight,:dataChannels]
        dataChannels=img.shape[2]
        print('shape img=',img.shape)
        #
        img = img.reshape(imgWidth,imgHeight,dataChannels,-1)  #adding channel dimension
        img = np.transpose(img,(2,0,1,3)) #reordering arrays
        
        #% predisct the segmanetation
        
        y_prob = model_trained.predict(img)
        mask_pred=np.argmax(y_prob,axis=3)
        
        #% plot the results 
        
        slic_disp=5
        #evaluate
        plot_slice_test(img[slic_disp,:,:], mask_pred[slic_disp,:,:])
        
        
    
        #% save the 

        img_nii = nib.Nifti1Image(mask_pred, np.eye(4))
        img_nii.get_data_dtype() == np.dtype(np.int16)
        img_nii.header.get_xyzt_units()
        nib.save(img_nii, save_nii+fname.replace('image','mask_predicted')+'.gz') 
        
#%% 
print('#######  THE END #########')

slic_disp=234
#evaluate
plot_slice_test(img[slic_disp,:,:], mask_pred[slic_disp,:,:])