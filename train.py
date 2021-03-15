# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 15:53:41 2021

@author: Abderrazak Chahid
"""


import sys
import os
import numpy as np
import pickle
from data_handler import *

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

from model import *
####

def diceCoef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def diceCoefLoss(y_true, y_pred):
    return (1-diceCoef(y_true, y_pred))

def jaccardDistance(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1-jac) * smooth

def f1Score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val


def plot_slice( img, mask, predicted_mask):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
        
    fig = plt.figure()
    ax1 = fig.add_subplot(131)
       
    ax1.imshow(img, cmap='gray')
    plt.title('Input image Mask')
    
    
    ax2 = fig.add_subplot(132) 
    ax2.imshow(mask)
    plt.title('True Mask')



    ax3 = fig.add_subplot(133) 
    ax3.imshow(predicted_mask)
    plt.title('predicted Mask')


def to_categorical_tensor( x3d, n_cls ) :
    batch_size, n_rows, n_cols,d = x3d.shape
    x1d = x3d.ravel()
    y1d = to_categorical( x1d, num_classes = n_cls )
    y4d = y1d.reshape( [ batch_size, n_rows, n_cols, n_cls ] )
    return y4d

def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  return pred_mask[0]

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()
  
  
def show_predictions(image, mask ):

      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
 

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
    
    
    
#%% //////////////////////////////////////////////////////////////////////////    
# root_folder='../'
# data_folder=root_folder+'/data/data0'
# # concat=np.concatenate((imgVol,imgVol),axis=2) 
##/////////////////////////////////////////////////////////////////////////////////////////
print('#############################################')
print('#  Training Unet for CT image segmentation  #')
print('#############################################')

imgWidth = 64*1
imgHeight = 64*1
nb_epochs=150

## dataset path file
path_CT_dataset='../data/CT_dataset.np'

# Data handler, load the data
print('---> Loading dataset from paths stored in :',path_CT_dataset)

mydata=CT_data(path_CT_dataset)
mydata.load_data()
#
img0, mask0= mydata.train_img, mydata.train_label
print('Training shape img=',img0.shape)
print('Training shape mask=',mask0.shape)

#mydata.print_paths()
#%%


# ## Load data
# pathImgs, pathMasks = '../data0/tr_im.nii', '../data0/tr_mask.nii' #enter the filepath of CT images and masks
# imgRaw, maskRaw = nib.load(pathImgs), nib.load(pathMasks) #loading dataset from original file type
# img0, mask0 = np.asanyarray(imgRaw.dataobj), np.asanyarray(maskRaw.dataobj) #converting dataset to numpy ndarrays


print('Training shape img=',img0.shape)
print('Training shape mask=',mask0.shape)
#%% Get the training data
dataChannels=img0.shape[2]
print('Dataset loaded ad reshaped [for computational reasons]')
x0=300
y0=300
img=img0[x0:x0+imgWidth,y0:y0+imgHeight,:dataChannels]
mask=mask0[x0:x0+imgWidth,y0:y0+imgHeight,:dataChannels]

print('shape img=',img.shape)
print('shape mask=',mask.shape)

img, mask = img.reshape(imgWidth,imgHeight,dataChannels,-1), mask.reshape(imgWidth,imgHeight,dataChannels,-1) #adding channel dimension
img, mask = np.transpose(img,(2,0,1,3)), np.transpose(mask,(2,0,1,3)) #reordering arrays

print('Training shape img=',img.shape)
print('Training shape mask=',mask.shape)
#%%
# mask[mask > 0] = 1 #binarizing masks
mask_old=mask
classes0=np.unique(mask)
num_classes = len(classes0)
new_class=np.asarray([i for i in range(num_classes)])


def convert_classes(x, old_class, new_class):
    for i in range(len(old_class)):
        x[x==old_class[i]] = new_class[i]
    
    return x

masks=convert_classes(mask_old, classes0, new_class)
  
print('new classss=', np.unique(masks))  
#%
mask=to_categorical_tensor( mask, num_classes )

trainImgs, testImgs, trainMasks, testMasks = train_test_split(img, mask, test_size = 0.2) #splitting the dataset
print('shape img=',img.shape)
print('shape mask=',mask.shape)
restore_label= tf.math.argmax(mask, axis=1)

#%%
print('Dataset split completed')
print('number of classes: ',len(np.unique(trainMasks)))
# print('number of classes: ',len(np.unique(mask0)))

print('Tensor sizes:')
print('Training images:'+str(trainImgs.shape)+'\nTest images:'+str(testImgs.shape)+'\nTest masks:'+str(trainMasks.shape)+'\nTest masks:'+str(testMasks.shape))

# input('break?')

dataGenArgs = dict(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1) #ImageDataGenerator arguments
imageGen, maskGen = keras.preprocessing.image.ImageDataGenerator(**dataGenArgs), keras.preprocessing.image.ImageDataGenerator(**dataGenArgs)

seed = 1
imageGen.fit(trainImgs, augment=True, seed=seed)
maskGen.fit(trainMasks, augment=True, seed=seed)

imageGenerator, maskGenerator = imageGen.flow(trainImgs, shuffle=False, batch_size = 8, seed=seed), maskGen.flow(trainMasks, shuffle=False, batch_size = 8, seed=seed)

# trainGenerator = zip(imageGenerator, maskGenerator)

trainGenerator = (pair for pair in zip(imageGenerator, maskGenerator))

# 
checkpoint=1
#%% learning / training
model = Unet(imgWidth, imgHeight, 1, num_classes)# keras.Model(inputs=[inputs], outputs=[outputs])
metrics = ['accuracy', diceCoef, jaccardDistance, f1Score]
optimizer_list=['adam',keras.optimizers.RMSprop(learning_rate=0.0001)]
loss_list=['binary_crossentropy', diceCoefLoss]

# model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001), loss=diceCoefLoss , metrics=metrics)
model.compile(optimizer='adam', loss='binary_crossentropy' , metrics=metrics)
model.summary()


#%% 

# for loop in range(2):
history = model.fit(trainImgs, trainMasks, validation_split=0.2, epochs=nb_epochs, batch_size=64, verbose=1)

## Save the trained model
results_path='../data/models/Unet_weights_sizeH'+str(imgHeight)+'_tag'+str(checkpoint)+'.h5'; 
model.save(results_path) ; checkpoint=checkpoint+1 #enter filepath for saving model weights

#% list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.figure(1)
plt.plot(history.history['jaccardDistance'])
plt.plot(history.history['val_jaccardDistance'])
plt.title('model jaccardDistance')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.figure(2)
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss (diceCoef)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.figure(3)
# summarize history for loss
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#%  prediction / deployment
## Evaluate performance
model.evaluate(testImgs, testMasks, batch_size = 8) #evaluate model

x=testImgs
ymask=testMasks

y=np.argmax(ymask,axis=3)  
#predicting 
y_prob = model.predict(x)
y_pred=np.argmax(y_prob,axis=3)

print('number of classes in mask: ',np.unique(y))
print('number of classes in prediction: ',np.unique(y_pred))
print('deploy mask size=',y.shape)
print('deploy prediction size=',y_pred.shape)


slic_disp=14
# sz=64
# print('sample mask=',y[slic_disp,:sz,:sz])
# print('sample predicted mask=',y_pred[slic_disp,:sz,:sz])

#evaluate
plot_slice(x[slic_disp,:,:], y[slic_disp,:,:], y_pred[slic_disp,:,:])


#%%
print('#######  THE END #########')

