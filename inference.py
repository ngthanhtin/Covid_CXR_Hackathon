from re import sub
import pandas as pd
import numpy as np
from glob import glob
import argparse
import os

import torch
from torch.utils.data import DataLoader

from models import CXRClassifier
from dataset import CXR_Dataset_Test, CXR_Dataset
from augmentation import get_augmentation
from config import config
from tqdm import tqdm
import gc
def inference(model, loader):
    preds = []
    names = []
    for batch_idx, (data, name) in tqdm(enumerate(loader), total=len(loader)):
        # data = data.to(config.device)
        data = data.to(config.device)
        output = model(data)
        preds += [output.data.cpu()]
        name = [x.split('/')[-1] for x in name]
        names += name
    preds = torch.cat(preds)

    return preds, names

def blended_cnn():
    """
    models: a list of model
    """
    path = '../TestSet_Gamma/'
    image_path = path + 'TestSet_Gamma/'

    # metadata_path = '../TestSet/testClinData.xls'
    # metadata_df = pd.read_excel(metadata_path)
    # metadata_df['ImageFile'] = metadata_df['ImageFile']
    
    # Create the pandas DataFrame
    submission = pd.DataFrame(columns = ['Name', 'Age'])
    
    mapping = {0: 'SEVERE', 1: 'MILD'}
    # selected_columns['Prognosis'] = selected_columns['Prognosis'].apply(lambda class_id: mapping[class_id]) 
    # true_targets = selected_columns['Prognosis'].to_list()
    # true_targets = np.array(true_targets)

    # for densenet121
    test_dataset = CXR_Dataset_Test(image_path, transform=get_augmentation('test', 300, 256))
    test_loader = DataLoader(dataset=test_dataset, batch_size=4,
                              shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    model = CXRClassifier(n_labels=config.N_CLASSES, model_name='densenet121').to(config.device)
    checkpoint = torch.load('./pretrained_models/densenet121_best_7346.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    densenet_preds, names = inference(model, test_loader)
    
    # for Swin224
    # test_dataset = CXR_Dataset(selected_columns, transform=get_augmentation('test', 256, 224))
    # test_loader = DataLoader(dataset=test_dataset, batch_size=4,
    #                           shuffle=False, num_workers=8, pin_memory=True)
    # swinnet_preds = inference(swinnet, test_loader)
    
    # predictions
    data_size = len(test_loader.dataset)
    preds_labels= torch.max(densenet_preds,dim=-1)[1].cpu().numpy()
    print(len(preds_labels))
    save_file = open("submission.txt", 'w')
    for i, label in enumerate(preds_labels):
        submission = submission.append({'ImageFile': names[i], 'Prognosis': mapping[label]}, ignore_index = True)
        save_file.write("{},{}\n".format(names[i], mapping[label]))
    submission.to_csv('submission.csv')
    save_file.close()

if __name__ == '__main__':
    blended_cnn()