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
import glob

preprocess=transforms.Compose([
            transforms.Resize((300,300)),
            transforms.CenterCrop(175),
            transforms.ToTensor()
        ])
def crop_with_argwhere(image):
    # Mask of non-black pixels (assuming image has a single channel).
    image = image[100:,100:]
    mask = image > 10
    
    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)
    
    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    
    # Get the contents of the bounding box.
    cropped = image[x0:x1, y0:y1]
    return cropped


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
        

        image = Image.open(image_name).convert('RGB')
        
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
#             if image2 is not None:
#                 image2 = self.transform(image2)
        
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


class CXR_Dataset_Visualization(Dataset):
    """
    Read images and corresponding labels.
    """
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir: path to image directory.
            transform: optional transform to be applied on a sample.
        """
        self.image_names = [img for img in glob.glob(os.path.join(image_dir, '*.png'),recursive=True)]
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its name
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        size = image.size
#         print(size)
        if self.transform is not None:
            image = self.transform(image)
        return image, image_name.split('/')[-1], size

    def __len__(self):
        return len(self.image_names)
