import sys
sys.path.insert(0, './')
import torch

from models import CXRClassifier
from dataset import CXR_Dataset

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import pandas as pd
from config import config

import torch
from torch.utils.data import DataLoader

from explainations.utils import PathExplainerTorch, monotonically_increasing_red
from augmentation import get_augmentation

## obviously you'll want to replace this with the path to your own saved model
model_path = config.path_model_pretrained + '_best.pt'

classifier = CXRClassifier(n_labels=config.N_CLASSES)
checkpoint = torch.load(model_path)
classifier.load_state_dict(checkpoint['model_state_dict'])
classifier.eval()

pet = PathExplainerTorch(classifier.cpu())


# dataset
path = '../TrainSet/'
image_path = path + 'TrainSet/'
metadata_path = path + 'trainClinData.xls'
metadata_df = pd.read_excel(metadata_path)
# for col_name in metadata_df.columns: 
#     print(col_name, metadata_df[col_name].count())
metadata_df['ImageFile'] = image_path+metadata_df['ImageFile']
selected_columns = metadata_df[['ImageFile', 'Prognosis']]
mapping = {'SEVERE': 0, 'MILD': 1}
selected_columns['Prognosis'] = selected_columns['Prognosis'].apply(lambda class_id: mapping[class_id]) 
image_df = selected_columns.copy()

train_df = image_df.sample(frac=0.8,random_state=200) #random state is a seed value
val_df = image_df.drop(train_df.index)
train_df = train_df.reset_index()
val_df = val_df.reset_index()

train_dataset = CXR_Dataset(train_df, transform=get_augmentation(phase='train'))
train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=8, pin_memory=True)

val_dataset = CXR_Dataset(val_df, transform=get_augmentation(phase='valid'))              
val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)


background_ds = torch.zeros(200,3,256,256)
for i,x in enumerate(train_dataset):
    background_ds[i,:,:,:] = x[0]
    if i == 199:
        break

x = val_dataset[6]

example = x[0].view(1,3,256,256)
example.requires_grad = True

label = x[1]

odds = np.exp(classifier(example)[0][1].detach().numpy().item())
probs = odds/(1+odds)

output_attribs = pet.attributions(example,background_ds,output_indices=torch.tensor([1]),use_expectation=True,num_samples = 200)
##
## for visualization purposes, we take the mean absolute EG values 
## (variable named "shaps" here because EG calculates an Aumann-Shapley value)
##
ma_shaps = output_attribs.abs().mean(0).mean(0).detach().numpy()

sb.set_style("white")
fig, (showcxr,heatmap) = plt.subplots(ncols=2,figsize=(14,5))
hmap = sb.heatmap(ma_shaps,
        cmap = monotonically_increasing_red(),
        linewidths=0,
        zorder = 2,
        vmax = np.percentile(ma_shaps.flatten(), 99.5) ## we clip attributions at 99.5th percentile
        )                                              ## to fix Coverage (see http://ceur-ws.org/Vol-2327/IUI19WS-ExSS2019-16.pdf)
cxr=example.detach().numpy().squeeze().transpose(1,2,0)    
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
cxr = std * cxr + mean
cxr = np.clip(cxr, 0, 1)

hmap.axis('off')

showcxr.imshow(cxr)
showcxr.axis('off')
fig.suptitle('EG Attributions, True Label: {}, Pred: {:0.4f}'.format(label[1],probs), fontsize=16)
plt.show()