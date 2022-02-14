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

    for batch_idx, (data, labels) in tqdm(enumerate(loader), total=len(loader)):
        # data = data.to(config.device)
        data = data.to(config.device)
        output = model(data)
        preds += [output.data.cpu()]

    preds = torch.cat(preds)
    
    return preds

def blended_cnn():
    # true_targets = [0,1,0,1,1,0,0,1,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,0]
    # true_targets = np.array(true_targets)
    """
    models: a list of model
    """
    path = '../TrainSet_Gamma/'
    image_path = path + 'TrainSet_Gamma/'

    metadata_path = '../TrainSet/trainClinData.xls'
    metadata_df = pd.read_excel(metadata_path)
    metadata_df['ImageFile'] = image_path+metadata_df['ImageFile']
    image_path = metadata_df['ImageFile']
    selected_columns = metadata_df[['ImageFile', 'Prognosis']]
    mapping = {'SEVERE': 0, 'MILD': 1}
    selected_columns['Prognosis'] = selected_columns['Prognosis'].apply(lambda class_id: mapping[class_id]) 
    true_targets = selected_columns['Prognosis'].to_list()
    true_targets = np.array(true_targets)

    # for densenet121
    test_dataset = CXR_Dataset(selected_columns, transform=get_augmentation('test', 300, 256))
    test_loader = DataLoader(dataset=test_dataset, batch_size=4,
                              shuffle=False, num_workers=8, pin_memory=True)
    densenet = CXRClassifier(n_labels=config.N_CLASSES, model_name='densenet121').to(config.device)
    checkpoint = torch.load('./pretrained_models/densenet121_best_7346.pt')
    densenet.load_state_dict(checkpoint['model_state_dict'])
    densenet.eval()
    swinnet = CXRClassifier(n_labels=config.N_CLASSES, model_name='swin224').to(config.device)
    checkpoint = torch.load('./pretrained_models/swin__7568.pt')
    swinnet.load_state_dict(checkpoint['model_state_dict'])
    swinnet.eval()
    densenet_preds = inference(densenet, test_loader)
    
    # for Swin224
    test_dataset = CXR_Dataset(selected_columns, transform=get_augmentation('test', 256, 224))
    test_loader = DataLoader(dataset=test_dataset, batch_size=4,
                              shuffle=False, num_workers=8, pin_memory=True)
    swinnet_preds = inference(swinnet, test_loader)
    
    # predictions
    preds_labels= torch.max(densenet_preds,dim=-1)[1].cpu().numpy()
    correct=float(np.sum(preds_labels==true_targets))
    print(correct)
    preds_labels= torch.max(swinnet_preds,dim=-1)[1].cpu().numpy()
    correct=float(np.sum(preds_labels==true_targets))
    print(correct)
    preds = 0.6*densenet_preds + 0.4*swinnet_preds
    preds_labels= torch.max(preds,dim=-1)[1].cpu().numpy()
    data_size = len(test_loader.dataset)
    correct=float(np.sum(preds_labels==true_targets))
    print(correct)
    

def ensemble_with_metadata():
    sample = pd.read_csv(os.path.join('../TrainSet/', 'sample_submission.csv'))

    cnn_pred = pd.read_csv('./blended_cnn.csv')
    meta_pred = pd.read_csv('./meta_pred.csv')

    sample['target'] = (cnn_pred['target'] * 0.9 + meta_pred['target'] * 0.1)

    # final submissions
    sample.to_csv('ensembled.csv', header=True, index=False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub-dir', type=str, default='./subs')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    blended_cnn()