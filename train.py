#!/usr/bin/env python
#
# train_covid.py
#
# Run ``python train_covid.py -h'' for information on using this script.
#

import os
import sys

import argparse
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc, f1_score, classification_report
from tqdm import tqdm
import random

import torch
from torch.utils.data import DataLoader

from models import CXRClassifier
from dataset import CXR_Dataset_Test, CXR_Dataset
from augmentation import get_augmentation
from config import config

import wandb

wandb.login()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set True to be faster
    print(f'Setting all seeds to be {seed} to reproduce...')
seed_everything(1024)

def train(model, loss_func, train_loader, optimizer, epoch, scheduler):
    model.train()
    running_loss = None
    for batch_idx, (data, labels) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data, labels = data.to(config.device), labels.to(config.device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(epoch, batch_idx, loss))
        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * 0.99 + loss.item()*0.01
    
    print("Epoch {} Train Loss: {:.4f}".format(epoch, running_loss))

    if scheduler is not None:
        scheduler.step()
    
    return running_loss

def valid(model, loss_func, val_loader, epoch):
    model.eval()
    loss_sum = 0.
    sample_num = 0
    correct=0.0
    val_data_size = len(val_loader.dataset)
    
    gt_global = torch.FloatTensor()
    pred_global = torch.FloatTensor()

    for batch_idx, (data, labels) in tqdm(enumerate(val_loader), total=len(val_loader)):
        if batch_idx % 2000 == 0:
            print('testing process:', batch_idx)
        data, labels = data.to(config.device), labels.to(config.device)
        output = model(data)
        loss = loss_func(output, labels)
        loss_sum += loss.item()*labels.shape[0]
        sample_num += labels.shape[0]

        pred= output.data.cpu()
        
        preds_labels= torch.max(pred,dim=-1)[1].cpu().numpy()
        gt_labels=torch.max(labels,dim=-1)[1].cpu().numpy()

        pred_global = torch.cat((pred_global,pred),0)
        gt_global = torch.cat((gt_global, labels.data.cpu()), 0)

        correct+=float(np.sum(preds_labels==gt_labels))

    print("Epoch {} Valid Loss: {:.4f}".format(epoch, loss_sum/sample_num))
    valid_loss = loss_sum/sample_num

    preds_global_labels=torch.max(pred_global,dim=-1)[1].cpu().numpy()
    gt_global_labels=torch.max(gt_global, dim=-1)[1].cpu().numpy()
    f1 = f1_score(gt_global_labels, preds_global_labels, average='weighted')
    tn, fp, fn, tp = confusion_matrix(gt_global_labels, preds_global_labels).ravel()
    recall = tp/(tp+fn)
    specificity = tn/(fp+tn)
    balanced_acc = (recall+specificity)/2
    # print(classification_report(gt_global_labels,preds_global_labels,labels=[0,1,2,3]))

    return correct/val_data_size, f1, recall, specificity, balanced_acc, valid_loss

def main():
    
    path = '../TrainSet_Gamma/'
    image_path = path + 'TrainSet_Gamma/'
    metadata_path = config.metadata_path
    metadata_df = pd.read_excel(metadata_path)
    metadata_df['ImageFile'] = image_path+metadata_df['ImageFile']
    
    selected_columns = metadata_df[['ImageFile', 'Prognosis']]
    mapping = {'SEVERE': 0, 'MILD': 1}
    selected_columns['Prognosis'] = selected_columns['Prognosis'].apply(lambda class_id: mapping[class_id]) 
    image_df = selected_columns.copy()
    
    gkf  = GroupKFold(n_splits = 5)
    image_df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(image_df, groups = image_df.ImageFile.tolist())):
        image_df.loc[val_idx, 'fold'] = fold
        
    val_df = image_df[image_df.fold==config.fold]
    train_df = image_df[image_df.fold!=config.fold]
    print(len(val_df), len(train_df))
    val_df = val_df.reset_index()
    train_df = train_df.reset_index()
    # Check label distribution
    print(val_df.Prognosis.value_counts())
    print(train_df.Prognosis.value_counts())


    train_dataset = CXR_Dataset(train_df, transform=get_augmentation('train', config.input_size, config.crop_size))
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = CXR_Dataset(val_df, transform=get_augmentation('valid', config.input_size, config.crop_size))              
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size//2,
                             shuffle=False, num_workers=8, pin_memory=True)

    print('********************load data succeed!********************')
    print('********************load model********************')

    classifier = CXRClassifier(n_labels=config.N_CLASSES, model_name=config.model_name).to(config.device)

    loss_func = torch.nn.BCEWithLogitsLoss()
    # loss_func = torch.nn.CrossEntropyLoss()

    params_group = [{'params': classifier.parameters(), 'lr':float(config.LR)},]


    if config.optimizer =='sgd':
        optimizer = torch.optim.SGD(params_group, lr=float(config.LR), weight_decay = config.weight_decay, momentum = 0.9, nesterov=True)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(params_group, lr=float(config.LR), weight_decay = config.weight_decay)
    elif config.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(params_group, lr=float(config.LR), alpha=0.9, weight_decay = config.weight_decay, momentum = 0.9)
    elif config.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params_group, lr=float(config.LR), weight_decay = config.weight_decay)

    if config.scheduler == 'steprl':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = config.lr_decay_step, gamma=config.lr_decay_gamma)
    elif config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.train_number_epochs - config.start_epoch - 1)


    if config.continue_train==True:
        print('loading status...')
        checkpoint = torch.load(config.resume_model)
        classifier.load_state_dict(checkpoint['model_state_dict'])
    
    classifier.to(config.device)

    best_acc = 0.0
    for epoch in range(config.start_epoch, config.train_number_epochs):
        print('*******training!*********')
        loss = train(classifier, loss_func, train_loader, optimizer, epoch, scheduler)
        
        print('*******testing!*********')
        acc, f1, recall, specificity, balanced_acc, valid_loss = valid(classifier, loss_func, val_loader, epoch)
        
        print("Epoch {}, acc: {:.4f}, f1_score: {:.4f}, recall: {:.4f}, specificity: {:.4f}, \
            balanced acc: {:.4f}".format(epoch, acc, f1, recall, specificity, balanced_acc)) 
        
        # save
        # if epoch % 10 == 0:
        #     torch.save({
        #             'model_state_dict': classifier.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'scheduler': scheduler,
        #             'epoch': epoch
        #                 }, config.path_model_pretrained+ '_last.pt')
        
        if balanced_acc > best_acc:
            best_acc = balanced_acc
            torch.save({
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler,
                    'epoch': epoch
                        }, config.path_model_pretrained+ '_best.pt')


        wandb.log({"Train Loss": loss})
        wandb.log({"Valid Loss": valid_loss})
        wandb.log({"Accuracy": acc})



if __name__ == "__main__":
    run = wandb.init(project='Covid2022', 
                 job_type='Train',
                 anonymous='must')
    main()
    run.finish()
