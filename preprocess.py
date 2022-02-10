import numpy as np
import os
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

def test_batch():
    # dataset
    path = '../TrainSet/'
    image_path = path + 'TrainSet/'
    metadata_path = path + 'trainClinData.xls'
    metadata_df = pd.read_excel(metadata_path)
    metadata_df['ImageFile'] = image_path+metadata_df['ImageFile']
    selected_columns = metadata_df[['ImageFile', 'Prognosis']]
    mapping = {'SEVERE': 0, 'MILD': 1}
    selected_columns['Prognosis'] = selected_columns['Prognosis'].apply(lambda class_id: mapping[class_id]) 
    image_df = selected_columns.copy()

    train_df = image_df.sample(frac=0.8,random_state=200) #random state is a seed value
    val_df = image_df.drop(train_df.index)
    train_df = train_df.reset_index()
    val_df = val_df.reset_index()

    aug = A.Compose([A.Resize(height=256, width=256), ToTensorV2()])
    batch_size = 5
    dataset = CXR_Dataset(train_df, transform=aug)
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)



    for batch_idx, (data, labels) in enumerate(loader):
        x = data
        example = x.view(batch_size,3,256,256).reshape(1,-1).numpy()

        gcn_numpy = np.asarray([global_contrast_normalization(x) for x in example]).reshape((batch_size,256,256,3))

        print('Before GCN Data Mean and STD')
        print(example.mean(-1).mean(),example.std(-1).mean() )
        showimages(example.reshape((batch_size,256,256,3)),coloums=5,row=1,col=True)

        print('After GCN Data Mean and STD - VIEWING IMAGE IS NORMALIZED')
        print(gcn_numpy.mean(axis=(1,2,3)).mean(),gcn_numpy.std(axis=(1,2,3)).mean() )
        gcn_numpy = (gcn_numpy - gcn_numpy.min(axis=(1,2))[:,None,None,:] )/(gcn_numpy.max(axis=(1,2))-gcn_numpy.min(axis=(1,2)))[:,None,None,:]
        showimages(gcn_numpy.reshape((batch_size,256,256,3)),coloums=5,row=1,col=True)

        exit()

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