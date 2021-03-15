# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:16:33 2021

@author: Science
"""

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


#%% defining U-Net
def Unet(imgWidth=512, imgHeight=512, imgChannels=1, num_classes=1):
    inputs = keras.layers.Input((imgWidth, imgHeight, imgChannels), name = 'input')
    c1 = keras.layers.Conv2D(16, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c1-1")(inputs)
    d1 = keras.layers.AlphaDropout(0.05, name = "d1")(c1)
    c1 = keras.layers.Conv2D(16, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c1-2")(d1)
    p1 = keras.layers.MaxPooling2D((2,2), name = "p1")(c1)
    
    c2 = keras.layers.Conv2D(32, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c2-1")(p1)
    d2 = keras.layers.AlphaDropout(0.05, name = "d2")(c2)
    c2 = keras.layers.Conv2D(32, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c2-2")(d2)
    p2 = keras.layers.MaxPooling2D((2,2), name = "p2")(c2)
    
    c3 = keras.layers.Conv2D(64, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c3-1")(p2)
    d3 = keras.layers.AlphaDropout(0, name = 'd3')(c3)
    c3 = keras.layers.Conv2D(64, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c3-2")(d3)
    p3 = keras.layers.MaxPooling2D((2,2), name = "p3")(c3)
    
    c4 = keras.layers.Conv2D(128, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c4-1")(p3)
    d4 = keras.layers.AlphaDropout(0.05, name = "d4")(c4)
    c4 = keras.layers.Conv2D(128, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c4-2")(d4)
    p4 = keras.layers.MaxPooling2D((2,2), name = "p4")(c4)
    
    c5 = keras.layers.Conv2D(256, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c5-1")(p4)
    d5 = keras.layers.AlphaDropout(0, name = "d5")(c5)
    c5 = keras.layers.Conv2D(256, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c5-2")(d5)
    u6 = keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same', name = "t1")(c5)
    
    u6 = keras.layers.concatenate([u6, c4], name = "cc1")
    c6 = keras.layers.Conv2D(128, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c6-1")(u6)
    d6 = keras.layers.AlphaDropout(0, name = "d6")(c6)
    c6 = keras.layers.Conv2D(128, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c6-2")(d6)
    u7 = keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same', name = "t2")(c6)
    
    u7 = keras.layers.concatenate([u7, c3], name = "cc2")
    c7 = keras.layers.Conv2D(64, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c7-1")(u7)
    d7 = keras.layers.AlphaDropout(0.05, name = "d7")(c7)
    c7 = keras.layers.Conv2D(64, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c7-2")(d7)
    u8 = keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same', name = "t3")(c7)
    
    u8 = keras.layers.concatenate([u8, c2], name = "cc3")
    c8 = keras.layers.Conv2D(32, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c8-1")(u8)
    d8 = keras.layers.AlphaDropout(0, name = "d8")(c8)
    c8 = keras.layers.Conv2D(32, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c8-2")(d8)
    u9 = keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same', name = "t4")(c8)
    
    u9 = keras.layers.concatenate([u9, c1], name = "cc4")
    c9 = keras.layers.Conv2D(16, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c9-1")(u9)
    d9 = keras.layers.AlphaDropout(0.05, name = "d9")(c9)
    c10 = keras.layers.Conv2D(16, (1,1), strides=(1,1), padding='same', name = "c10")(d9)
    
    outputs = keras.layers.Conv2D(num_classes,(1,1), activation='sigmoid', name = "output")(c10)
    # outputs = keras.layers.Conv2D(num_classes,(1,1), activation='softmax', name = "output")(c10)
    
    return keras.Model(inputs=[inputs], outputs=[outputs])
