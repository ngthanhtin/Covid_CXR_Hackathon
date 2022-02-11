# encoding: utf-8

"""
Read images and corresponding labels.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import os
import random
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd
import glob

class CXR_Dataset_Test(Dataset):
    def __init__(self, df, transform=None, crop=False):
        self.NUM_CLASSES = 2
        self.crop = crop
        # Set of images for each class
        self.image_names = df['ImageFile']
        self.labels = df['Prognosis']

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        def __one_hot_encode(l):
            v = [0] * self.NUM_CLASSES
            v[l] = 1
            return v

        image_name, label = self.image_names[index], self.labels[index]
        label = __one_hot_encode(label)
        name = image_name.split("/")[-1]
        
#         image2=None
        
#         if self.crop:
#             try:
#                 src = ImageOps.grayscale(Image.open(image_name).convert('RGB'))
#                 tmp = np.array(src).astype(np.uint8)
# #                 tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
#             except cv2.error:
#                 print(image_name)
#             newcrop = crop_with_argwhere(tmp)
#             crop_rgb = ImageOps.grayscale(Image.fromarray(cv2.cvtColor(newcrop,cv2.COLOR_GRAY2RGB)))
#             img = preprocess(crop_rgb)
#             arr = (img.squeeze(0).numpy()*255).astype(np.uint8)
#             recrop = Image.fromarray(arr).resize((480,480))
#             image2 = recrop.convert('RGB')
        

        # image = Image.open(image_name).convert('RGB')
        image = cv2.imread(image_name)
        if image is None:
            print(image_name)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            image = self.transform(image)
#             if image2 is not None:
#                 image2 = self.transform(image2)
        
        
#         print(image_name.split("/")[-1])
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


class CXR_Dataset(Dataset):
    def __init__(self, df, transform=None, crop=False):
        self.NUM_CLASSES = 2
        self.crop = crop
        # Set of images for each class
        self.image_names = df['ImageFile']
        self.labels = df['Prognosis']
               
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """

        def __one_hot_encode(l):
            v = [0] * self.NUM_CLASSES
            v[l] = 1
            return v

        image_name = None

        image_name, label = self.image_names[index], self.labels[index]
        label=__one_hot_encode(label)
        
        assert image_name is not None
        
        image = cv2.imread(image_name)
        if image is None:
            print(image_name)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)

#     def loss(self, output, target, gamma):
#         """
#         Binary weighted focal loss for each class
#         """
#         weight_plus = torch.autograd.Variable(self.loss_weight_plus.repeat(
#             1, target.size(0)).view(-1, self.loss_weight_plus.size(1)).cuda())
#         weight_neg = torch.autograd.Variable(self.loss_weight_minus.repeat(
#             1, target.size(0)).view(-1, self.loss_weight_minus.size(1)).cuda())

#         loss = output
#         pmask = (target >= 0.5).data
#         nmask = (target < 0.5).data

#         epsilon = 1e-15
#         loss[pmask] = torch.pow(1-loss[pmask], gamma) * \
#             (loss[pmask] + epsilon).log() * weight_plus[pmask]
#         loss[nmask] = torch.pow(loss[nmask], gamma) * \
#             (1-loss[nmask] + epsilon).log() * weight_plus[nmask]
#         loss = -loss.sum()
#         return loss


class MetaCXR_Dataset(Dataset):
    def __init__(self, df, transform=None, crop=False):
        self.NUM_CLASSES = 2
        self.crop = crop
        # Set of images for each class
        self.image_names = df['ImageFile']
        self.labels = df['Prognosis']
        self.metadata_df = get_related_features_df(df)

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """

        def __one_hot_encode(l):
            v = [0] * self.NUM_CLASSES
            v[l] = 1
            return v

        image_name = None

        image_name, label = self.image_names[index], self.labels[index]
        label=__one_hot_encode(label)
        
        assert image_name is not None
        
        image = cv2.imread(image_name)
        if image is None:
            print(image_name)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, torch.tensor(self.metadata_df.iloc[index]).float(), torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


def get_related_features_df(df):

    # related features as mentioned in the challenge
    related_features = ['Age', 'Sex', 'Temp_C', 'Cough', 'DifficultyInBreathing', 'WBC', 'CRP', 'Fibrinogen', \
        'LDH', 'D_dimer', 'Ox_percentage', 'PaO2', 'SaO2', 'pH', 'CardiovascularDisease', 'RespiratoryFailure']
    redundant_features = []

    #dop unecessary columns
    # metadata_df = metadata_df.drop(['Row_number', 'ImageFile', 'Hospital'], axis=1)
    for col_name in df.columns: 
        if col_name not in related_features:
            redundant_features.append(col_name)
    df = df.drop(redundant_features, axis=1)

    #check nan
    for col_name in df.columns: 
        print(col_name + ": ", df[col_name].isnull().values.any())
    #fill nan
    df['Age'] = df['Age'].fillna(0)
    df['Temp_C'] = df['Temp_C'].fillna(0)
    df['Cough'] = df['Cough'].fillna(0)
    df['DifficultyInBreathing'] = df['DifficultyInBreathing'].fillna(0)
    df['WBC'] = df['WBC'].fillna(0)
    df['CRP'] = df['CRP'].fillna(0)
    df['Fibrinogen'] = df['Fibrinogen'].fillna(0)
    df['LDH'] = df['LDH'].fillna(0)
    df['D_dimer'] = df['D_dimer'].fillna(0)
    df['Ox_percentage'] = df['Ox_percentage'].fillna(0)
    df['PaO2'] = df['PaO2'].fillna(0)
    df['SaO2'] = df['SaO2'].fillna(0)
    df['pH'] = df['pH'].fillna(0)
    df['CardiovascularDisease'] = df['CardiovascularDisease'].fillna(0)
    df['RespiratoryFailure'] = df['RespiratoryFailure'].fillna(0)

    return df