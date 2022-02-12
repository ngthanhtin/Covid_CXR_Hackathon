import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

from skimage import util 
from skimage.transform import resize
from skimage.io import imread
import cv2
from os import listdir
from os.path import isfile, join
from config import config

import torch
from torch.utils.data import DataLoader
from augmentation import get_augmentation
from models import CXRClassifier
from dataset import CXR_Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

def showimages(x,coloums=30,row=3,col=False):
    fig=plt.figure(figsize=(30, 3))
    columns = coloums; rows = row
    for i in range(1, columns*rows +1):
        fig.add_subplot(rows, columns, i)
        if col: 
            plt.imshow(np.squeeze(x[i-1]))
            # plt.imshow(x[0])
        else:   
            plt.imshow(np.squeeze(x[i-1]),cmap='gray')
        plt.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False, labeltop=False, labelright=False, labelbottom=False)
        plt.grid(False)
    plt.show()

# Modified code from: https://datascience.stackexchange.com/questions/15110/how-to-implement-global-contrast-normalization-in-python
def global_contrast_normalization(image):
    XX = image
    # replacement for the loop
    X_average = np.mean(XX)
    XX = XX - X_average
    
    ss   = 1.0
    lmda = 10.
    # `su` is here the mean, instead of the sum
    contrast = np.sqrt(lmda + np.mean(XX**2)).astype(np.float64)
    
    if contrast > 1e-8:
        XX = ss * XX / contrast
    else:
        XX = ss * XX 

    return XX

def rotateImage(image, angle):
    row,col = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

def randomZoom(image):
    x=random.randint(30,140)
    y=random.randint(0,20)
    crop_img = image[x:x+224, y:y+224]
    crop_img = cv2.resize(crop_img, (224, 224))
    return crop_img

def hist_rotate_zoom_aug():
    save_path = '/home/hci-a4000/TIN/covid2022/TrainSet_Preprocessed/TrainSet_Preprocessed/'
    ori_data_path = '/home/hci-a4000/TIN/covid2022/TrainSet/TrainSet/'

    
    onlyfiles = [f for f in listdir(ori_data_path) if isfile(join(ori_data_path, f))]
    for f in onlyfiles:
        img_name = f
        img_path = ori_data_path + f
        img = cv2.imread(img_path)
    
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_arr = cv2.resize(gray, (config.input_size, config.input_size))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(resized_arr)
        
        aug_img=rotateImage(equalized,random.randint(0,60) )
        if random.randint(0,1)==1:
            aug_img=randomZoom(aug_img)
    
    cv2.imwrite(save_path + img_name, aug_img)

def test_single_image():
    save_path = '/home/hci-a4000/TIN/covid2022/TrainSet_Preprocessed/TrainSet_Preprocessed/'
    ori_data_path = '/home/hci-a4000/TIN/covid2022/TrainSet/TrainSet/'

    
    onlyfiles = [f for f in listdir(ori_data_path) if isfile(join(ori_data_path, f))]
    for f in onlyfiles:
        img_name = f
        img_path = ori_data_path + f
        example = cv2.imread(img_path)
        # cv2.imshow("a", example)
        # cv2.waitKey(0)
        height, width = example.shape[:2]
        example = np.expand_dims(example, 0)
        example = example.reshape(1, -1)

        gcn_numpy = global_contrast_normalization(example).reshape((height,width,3))
        
        print('Before GCN Data Mean and STD')
        print(example.mean(-1).mean(),example.std(-1).mean() )
        example = example.reshape((height,width,3))
        # cv2.imshow("a1", example)
        # cv2.waitKey(0)

        print('After GCN Data Mean and STD - VIEWING IMAGE IS NORMALIZED')
        print(gcn_numpy.mean(axis=(0,1,2)).mean(),gcn_numpy.std(axis=(0,1,2)).mean() )
        gcn_numpy = (gcn_numpy - gcn_numpy.min(axis=(0,1))[None,None,:] )/(gcn_numpy.max(axis=(0,1))-gcn_numpy.min(axis=(0,1)))[None,None,:]
        # cv2.imshow("a2", gcn_numpy)
        # cv2.waitKey(0)
        
        cv2.imwrite(save_path + img_name, 255*gcn_numpy)
        

if __name__ == '__main__':
    test_single_image()