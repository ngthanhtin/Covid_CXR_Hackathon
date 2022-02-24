import sys
import os
from turtle import up
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import CXR_Dataset_Test, CXR_Dataset, CXR_Dataset_Visualization
from PIL import Image
from pprint import pprint
from augmentation import get_augmentation


# from covidaid_v2 import CovidAID, Fusion_Branch
from model import CovidAidAttend, Fusion_Branch
import argparse
from tqdm import tqdm
import cv2

import json
import datetime
# from sklearn.metrics import 
import time
from skimage.measure import label


import matplotlib
import matplotlib.pyplot as plt

from collections.abc import Sequence, Iterable
import pandas as pd
from skimage.draw import rectangle_perimeter, set_color
import types

from utils import plot_confusion_matrix, compute_AUC_scores, plot_ROC_curve
from augmentation import RandomAffine

USE_GPU = torch.cuda.is_available()

if USE_GPU:
    print("Using GPU..")

save_model_name='AGCNN'

LR_G = 1e-4
LR_L = 1e-4
LR_F = 1e-5

def Attention_gen_patchs(ori_image, fm_cuda,mode='normal'):
    
    # fm => mask =>(+ ori-img) => crop = patchs
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    minMaxList=[]
    heatMapList=[]
    
    if USE_GPU:
        feature_conv = fm_cuda.data.cpu().numpy()
    else:
        feature_conv = fm_cuda.data.numpy()

    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape
    
    if USE_GPU:
        patchs_cuda = torch.FloatTensor().cuda()
    else:
        patchs_cuda = torch.FloatTensor()

    for i in range(0, bz):
        feature = feature_conv[i]
        cam = feature.reshape((nc, h*w))
        cam = cam.sum(axis=0)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        upsampled_img = cv2.resize(cam_img, size_upsample)
        # upsampled_img = np.expand_dims(upsampled_img, axis=2)

        # cv2.imshow("a", upsampled_img)
        # cv2.waitKey(0)
        # exit()
        
        heatmap_bin = binImage(cv2.resize(cam_img, size_upsample))
        heatmap_maxconn = selectMaxConnect(heatmap_bin)
        heatmap_mask = heatmap_bin * heatmap_maxconn
        
        ############################
        
        heatMapList.append(heatmap_mask)
        
        
        ind = np.argwhere(heatmap_mask != 0)
#         print(len(ind[:,1]))
        
        if mode=='visualize':
#             print("Extracting Patch Dimensions for visualization")
            minh = min(np.sort(ind[:, 0])[3000:])
            minw = min(np.sort(ind[:, 1])[3000:])
            maxh = max(np.sort(ind[:, 0])[:-3000])
            maxw = max(np.sort(ind[:, 1])[:-3000])
        else:
            minh = min(ind[:, 0])
            minw = min(ind[:, 1])
            maxh = max(ind[:, 0])
            maxw = max(ind[:, 1])

        
        # to ori image
        image = ori_image[i].numpy().reshape(224, 224, 3)
        image = image[int(224*0.334):int(224*0.667),
                      int(224*0.334):int(224*0.667), :]
        
        # plt.imshow(image)

        image = cv2.resize(image, size_upsample)
#         plt.imshow(image)
        image_crop =(image[minh:maxh, minw:maxw, :]*std+mean) * 256
        
#         print(image_crop)
        
        # because image was normalized before
        
#         plt.imshow(image_crop)
        
        
        image_crop = preprocess(Image.fromarray(
            image_crop.astype('uint8')).convert('RGB'))
        
        
        # tensor_imshow(image_crop)
        
        minMaxList.append((minh,minw,maxh,maxw))
        
        if USE_GPU:
            img_variable = image_crop.view(3, 224, 224).unsqueeze(0).cuda()
        else:
            img_variable =image_crop.view(3, 224, 224).unsqueeze(0)
        
#         print(img_variable.shape)
        patchs_cuda = torch.cat((patchs_cuda, img_variable),0)

    # plt.show()
    heatMapList[0] = np.expand_dims(heatMapList[0], axis=2)
    
    # img = Image.fromarray(heatMapList[0], 'RGB')
    # img.show()
    # exit()
    return torch.autograd.Variable(patchs_cuda), minMaxList



def binImage(heatmap):
    _, heatmap_bin = cv2.threshold(
        heatmap, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # t in the paper
    #_, heatmap_bin = cv2.threshold(heatmap , 178 , 255 , cv2.THRESH_BINARY)
    return heatmap_bin


def selectMaxConnect(heatmap):
    labeled_img, num = label(heatmap, connectivity=2,
                             background=0, return_num=True)
    max_label = 0
    max_num = 0
    for i in range(1, num+1):
        if np.sum(labeled_img == i) > max_num:
            max_num = np.sum(labeled_img == i)
            max_label = i
    lcc = (labeled_img == max_label)
    if max_num == 0:
        lcc = (labeled_img == -1)
    lcc = lcc + 0
    return lcc


def train(CKPT_PATH_INIT='init',CKPT_PATH_G=None, CKPT_PATH_L=None, CKPT_PATH_F=None,
          epochs=50, batch_size=64, logging=True, 
          save_dir=None,combine_metadata=False, freeze=False):
    print('********************load data********************')
    if combine_metadata:
        N_CLASSES = 3
        CLASSES = ["Normal", "Pneumonia", "Covid"]
    else:
        N_CLASSES= 2
        CLASSES=["SEVERE","MILD"]
    
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    path = '../TrainSet_Gamma/'
    image_path = path + 'TrainSet_Gamma/'
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
    
    # transform=transforms.Compose([
    #                                      transforms.Resize(256),
    #                                      RandomAffine(30, scale=(0.8,1.2), shear=[-15,15,-15,15]),
    #                                      transforms.CenterCrop(224),
    #                                      transforms.ToTensor(),
    #                                      normalize,
    #                                  ])


    train_dataset = CXR_Dataset(train_df, transform=get_augmentation('train', 256,224))
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)


    # transform=transforms.Compose([
    #                                     transforms.Resize(256),
    #                                     transforms.CenterCrop(224),
    #                                     transforms.ToTensor(),
    #                                     normalize,
                                    # ])

    val_dataset = CXR_Dataset(val_df, transform=get_augmentation('valid', 256, 224))
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=8, pin_memory=True)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if USE_GPU:
        Global_Branch_model = CovidAidAttend(combine_metadata).cuda()
        Local_Branch_model = CovidAidAttend(combine_metadata).cuda()
        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES).cuda()
    else:
        Global_Branch_model = CovidAidAttend(combine_metadata)
        Local_Branch_model = CovidAidAttend(combine_metadata)
        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES)
    
#     print("Existence of CKPT_INIT:{}".format(os.path.exists(CKPT_PATH_INIT)))
    
    if os.path.exists(CKPT_PATH_INIT):
        print("=> loading Initial checkpoint")
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_INIT)
        else:
            checkpoint = torch.load(CKPT_PATH_INIT,map_location='cpu')
        Global_Branch_model.load_state_dict(checkpoint)
        Local_Branch_model.load_state_dict(checkpoint)
        print("=> Model weights initialized from CheXNet")

    else:
        print("=> Model training will be resumed")

        if os.path.exists(CKPT_PATH_G):
            if USE_GPU:
                checkpoint = torch.load(CKPT_PATH_G)
            else:
                checkpoint = torch.load(CKPT_PATH_G,map_location='cpu')
            Global_Branch_model.load_state_dict(checkpoint)
            print("=> loaded Global_Branch_model checkpoint")

        if os.path.exists(CKPT_PATH_L):
            if USE_GPU:
                checkpoint = torch.load(CKPT_PATH_L)
            else:
                checkpoint = torch.load(CKPT_PATH_L,map_location='cpu')

            Local_Branch_model.load_state_dict(checkpoint)
            print("=> loaded Local_Branch_model checkpoint")

        if os.path.exists(CKPT_PATH_F):
            if USE_GPU:
                checkpoint = torch.load(CKPT_PATH_F)
            else:
                checkpoint = torch.load(CKPT_PATH_F,map_location='cpu')

            Fusion_Branch_model.load_state_dict(checkpoint)
            print("=> loaded Fusion_Branch_model checkpoint")

    cudnn.benchmark = True
    criterion = nn.BCELoss()
    
    if freeze:
        print ("Freezing feature layers")
        for param in Global_Branch_model.densenet121.features.parameters():
            param.requires_grad = False
        for param in Local_Branch_model.densenet121.features.parameters():
            param.requires_grad = False
        
    
    optimizer_global = optim.Adam(Global_Branch_model.parameters(
    ), lr=LR_G, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    

    optimizer_local = optim.Adam(Local_Branch_model.parameters(
    ), lr=LR_L, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    


    optimizer_fusion = optim.Adam(Fusion_Branch_model.parameters(
    ), lr=LR_F, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    

    print('********************load model succeed!********************')
    

    print('********************begin training!********************')
    for epoch in range(epochs):
#         Global_Branch_model.train()
#         Local_Branch_model.train()
#         Fusion_Branch_model.train()
        since = time.time()
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        # set the mode of model
#         lr_scheduler_global.step()  # about lr and gamma
#         lr_scheduler_local.step()
#         lr_scheduler_fusion.step()
        Global_Branch_model.train()  # set model to training mode
        Local_Branch_model.train()
        Fusion_Branch_model.train()

        running_loss = 0.0
        # Iterate over data
        for i, (input_, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
            
            if USE_GPU:
                input_var = torch.autograd.Variable(input_.cuda())
                target_var = torch.autograd.Variable(target.cuda())
            else:
                input_var = torch.autograd.Variable(input_)
                target_var = torch.autograd.Variable(target)
            
            optimizer_global.zero_grad()
            optimizer_local.zero_grad()
            optimizer_fusion.zero_grad()

            # compute output
            output_global, fm_global, pool_global = Global_Branch_model(
                input_var)

            patchs_var,_ = Attention_gen_patchs(input_, fm_global)

            output_local, _, pool_local = Local_Branch_model(patchs_var)
            # print(fusion_var.shape)
            output_fusion = Fusion_Branch_model(pool_global.data, pool_local.data)
            #

            # loss
            loss1 = criterion(output_global, target_var)
            loss2 = criterion(output_local, target_var)
            loss3 = criterion(output_fusion, target_var)
            #

            loss = loss1*0.4 + loss2*0.4 + loss3*0.2

            if (i % 100) == 0:
                print('step: %5d total_loss: %.3f loss_1: %.3f loss_2: %.3f loss_3: %.3f' %(
                      i, loss.data.cpu().numpy(), loss1.data.cpu().numpy(), loss2.data.cpu().numpy(),                                                 loss3.data.cpu().numpy()))

            loss.backward()
            optimizer_global.step()
            optimizer_local.step()
            optimizer_fusion.step()

            # print(loss.data.item())
            running_loss += loss.data.cpu().numpy()
            del input_var, target_var
            # break
            '''
            if i == 40:
                print('break')
                break
            '''
        
        if USE_GPU:
            torch.cuda.empty_cache()
        
        epoch_loss = float(running_loss) / float(i)
        print(' Epoch over  Loss: {:.5f}'.format(epoch_loss))

        print('*******testing!*********')
        val_metrics=test(Global_Branch_model, Local_Branch_model,
                         Fusion_Branch_model, val_loader,len(val_dataset),'val')
        
        timestamp = str(datetime.datetime.now()).split('.')[0]
        log= json.dumps({
            'timestamp':timestamp,
            'epoch': epoch+1,
            'train_loss': float('%.5f' % epoch_loss),
            'val_acc_G': float('%.5f' % val_metrics['valG']),
            'val_acc_L': float('%.5f' % val_metrics['valL']),
            'val_acc_F': float('%.5f' % val_metrics['valF'])
        })
        
        if logging:
            print(log)
            
        logFile= os.path.join(save_dir,'train.log')
        if logFile is not None:
            with open(logFile,'a') as f:
                f.write("{}\n".format(log))

        # break

        # save
        if epoch % 1 == 0:
            save_path = save_dir
            torch.save(Global_Branch_model.state_dict(), os.path.join(save_path,
                       save_model_name+'_Global'+'_epoch_'+str(epoch)+'.pth'))
            print('Global_Branch_model already save!')
            torch.save(Local_Branch_model.state_dict(), os.path.join(save_path,
                       save_model_name+'_Local'+'_epoch_'+str(epoch)+'.pth'))
            print('Local_Branch_model already save!')
            torch.save(Fusion_Branch_model.state_dict(), os.path.join(save_path,
                       save_model_name+'_Fusion'+'_epoch_'+str(epoch)+'.pth'))
            print('Fusion_Branch_model already save!')

        time_elapsed = time.time() - since
        print('Training one epoch complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        
def test(model_global, model_local, model_fusion, test_loader,val_size,mode='val',
         cm_path='cm',roc_path='roc',combine_pneumonia=True, binary_eval=False):
    
    if combine_pneumonia:
        N_CLASSES = 3
        CLASSES = ["Normal", "Pneumonia", "Covid"]
    else:
        N_CLASSES= 2
        CLASSES=["SEVERE", "MILD"]

    # switch to evaluate mode
    BIN_Classes=["SEVERE","MILD"]
    if mode=='val':
        
        model_global.eval()
        model_local.eval()
        model_fusion.eval()
        cudnn.benchmark = True
        global_correct=0.0
        local_correct=0.0
        fusion_correct=0.0

        for i, (inp, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if i % 2000 == 0:
                print('testing process:', i)
#         if USE_GPU:
#             target = target.cuda()
            

            gt = target
            if USE_GPU:
                input_var = torch.autograd.Variable(inp.cuda())
            else:
                input_var = torch.autograd.Variable(inp)

            #output = model_global(input_var)

            output_global, fm_global, pool_global = model_global(input_var)

            patchs_var,_ = Attention_gen_patchs(inp, fm_global)

            output_local, _, pool_local = model_local(patchs_var)

            output_fusion = model_fusion(pool_global.data, pool_local.data)

            pred_global = output_global.data.cpu()
            pred_local = output_local.data.cpu()
            pred_fusion = output_fusion.data.cpu()
        
            preds_global_labels= torch.max(pred_global,dim=-1)[1].numpy()
            preds_local_labels= torch.max(pred_local,dim=-1)[1].numpy()
            preds_fusion_labels=torch.max(pred_fusion,dim=-1)[1].numpy()
            gt_labels=torch.max(gt,dim=-1)[1].numpy()
        
            global_correct+=float(np.sum(preds_global_labels==gt_labels))
            local_correct+=float(np.sum(preds_local_labels==gt_labels))
            fusion_correct+=float(np.sum(preds_fusion_labels==gt_labels))

            del input_var
    

        if USE_GPU:
            torch.cuda.empty_cache()
    
        global_acc=global_correct/val_size
    
        local_acc=local_correct/val_size
    
        fusion_acc=fusion_correct/val_size
    
        val_metrics = {'valG':global_acc, 'valL':local_acc, 'valF': fusion_acc}
    
        return val_metrics
    
    else:
        gt = torch.FloatTensor()
        pred_global = torch.FloatTensor()
        pred_local = torch.FloatTensor()
        pred_fusion = torch.FloatTensor()
        
        model_global.eval()
        model_local.eval()
        model_fusion.eval()
        cudnn.benchmark = True
        
        image_name_list=[]
        
        for i, (inp, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if i % 2000 == 0:
                print('testing process:', i)
#         if USE_GPU:
#             target = target.cuda()
            
#             image_name = list(image_name)
#             image_name_list+=image_name
            gt = torch.cat((gt,target),0)
            if USE_GPU:
                input_var = torch.autograd.Variable(inp.cuda())
            else:
                input_var = torch.autograd.Variable(inp)

            #output = model_global(input_var)

            output_global, fm_global, pool_global = model_global(input_var)

            patchs_var,_ = Attention_gen_patchs(inp, fm_global)

            output_local, _, pool_local = model_local(patchs_var)

            output_fusion = model_fusion(pool_global.data, pool_local.data)
            
            pred_global = torch.cat((pred_global,output_global.data.cpu()),0)
            pred_local = torch.cat((pred_local,output_local.data.cpu()),0)
            pred_fusion = torch.cat((pred_fusion,output_fusion.data.cpu()),0)
            
            del input_var, output_global, fm_global, pool_global, patchs_var, output_local, pool_local

        if USE_GPU:
            torch.cuda.empty_cache()
        
#         print("Image Name Length:",len(image_name_list))
        print("GT Shape:", gt.shape)
        print("Global Shape:" ,pred_global.shape)
        print("Local: " ,pred_local.shape)
        print("Fusion:" ,pred_fusion.shape)
        
        if binary_eval:
            pred_global = torch.from_numpy(pred_global.numpy())
            pred_local = torch.from_numpy(pred_local.numpy())
            pred_fusion = torch.from_numpy(pred_fusion.numpy())
            gt = torch.from_numpy(gt.numpy())

        if binary_eval:
            aucGlobal = compute_AUC_scores(gt.numpy(),pred_global.numpy(),BIN_Classes)
            aucLocal = compute_AUC_scores(gt.numpy(),pred_local.numpy(),BIN_Classes)
            aucFusion= compute_AUC_scores(gt.numpy(),pred_fusion.numpy(),BIN_Classes)
        else:
            aucGlobal = compute_AUC_scores(gt.numpy(),pred_global.numpy(),CLASSES)
            aucLocal = compute_AUC_scores(gt.numpy(),pred_local.numpy(),CLASSES)
            aucFusion= compute_AUC_scores(gt.numpy(),pred_fusion.numpy(),CLASSES)
            
        
        CM_GLOBAL='%s_global'%cm_path
        CM_LOCAL='%s_local'%cm_path
        CM_FUSION='%s_fusion'%cm_path
        
        METRICS_GLOBAL='%s_metrics.txt'%CM_GLOBAL
        METRICS_LOCAL='%s_metrics.txt'%CM_LOCAL
        METRICS_FUSION='%s_metrics.txt'%CM_FUSION

        with open(METRICS_GLOBAL,'w') as file:
            file.write('The average AUROC is {auc_avg:.4f} \n'.format(auc_avg=aucGlobal[0]))
            for data in aucGlobal[1:]:
                file.write('The AUROC of {0:} is {1:.4f} \n'.format(data[0], data[1]))
                
        with open(METRICS_LOCAL,'w') as file:
            file.write('The average AUROC is {auc_avg:.4f} \n'.format(auc_avg=aucLocal[0]))
            for data in aucLocal[1:]:
                file.write('The AUROC of {0:} is {1:.4f} \n'.format(data[0], data[1]))
                
        with open(METRICS_FUSION,'w') as file:
            file.write('The average AUROC is {auc_avg:.4f} \n'.format(auc_avg=aucFusion[0]))
            for data in aucFusion[1:]:
                file.write('The AUROC of {0:} is {1:.4f} \n'.format(data[0], data[1]))
                
        ROC_GLOBAL='%s_global'%roc_path
        ROC_LOCAL='%s_local'%roc_path
        ROC_FUSION='%s_fusion'%roc_path
        
        if binary_eval:
            plot_ROC_curve(gt.numpy(),pred_global.numpy(),BIN_Classes,ROC_GLOBAL)
            plot_ROC_curve(gt.numpy(),pred_local.numpy(),BIN_Classes,ROC_LOCAL)
            plot_ROC_curve(gt.numpy(),pred_fusion.numpy(),BIN_Classes,ROC_FUSION)
        else:
            plot_ROC_curve(gt.numpy(),pred_global.numpy(),CLASSES,ROC_GLOBAL)
            plot_ROC_curve(gt.numpy(),pred_local.numpy(),CLASSES,ROC_LOCAL)
            plot_ROC_curve(gt.numpy(),pred_fusion.numpy(),CLASSES,ROC_FUSION)
            
        
        
        preds_global_labels=torch.max(pred_global,dim=-1)[1].numpy()
        preds_local_labels=torch.max(pred_local,dim=-1)[1].numpy()
        preds_fusion_labels=torch.max(pred_fusion,dim=-1)[1].numpy()
        gt_labels=torch.max(gt,dim=-1)[1].numpy()
        
#         preds_new_global = labelConverter(preds_global_labels)
#         preds_new_local = labelConverter(preds_local_labels)
#         preds_new_fusion = labelConverter(preds_fusion_labels)
#         gt_new = labelConverter(gt_labels)
        
        if binary_eval:
            plot_confusion_matrix(gt_labels, preds_global_labels,BIN_Classes,CM_GLOBAL)
            plot_confusion_matrix(gt_labels, preds_local_labels,BIN_Classes,CM_LOCAL)
            plot_confusion_matrix(gt_labels, preds_fusion_labels,BIN_Classes,CM_FUSION)
        else:
            plot_confusion_matrix(gt_labels, preds_global_labels,CLASSES,CM_GLOBAL)
            plot_confusion_matrix(gt_labels, preds_local_labels,CLASSES,CM_LOCAL)
            plot_confusion_matrix(gt_labels, preds_fusion_labels,CLASSES,CM_FUSION)


        



def evaluate(test_list, CKPT_PATH_G=None, CKPT_PATH_L=None, CKPT_PATH_F=None,
             cm_path=None,roc_path=None,bs=16,combine_pneumonia=True):
    
    if combine_pneumonia:
        N_CLASSES = 3
        CLASSES = ["Normal", "Pneumonia", "Covid"]
    else:
        N_CLASSES= 4
        CLASSES=["Normal","Bacterial","Viral","Covid"]
    
    
    print('********************load data********************')
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    test_dataset = CXR_Dataset_Test(image_list_file=test_list,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize,
                                    ]),
                                       combine_pneumonia=combine_pneumonia,crop=True)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs,
                             shuffle=False, num_workers=4, pin_memory=True)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if USE_GPU:
        Global_Branch_model = CovidAidAttend(combine_pneumonia).cuda()
        Local_Branch_model = CovidAidAttend(combine_pneumonia).cuda()
#         Crop_model = CovidAidAttend(combine_pneumonia).cuda()

        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES).cuda()
    else:
        Global_Branch_model = CovidAidAttend(combine_pneumonia)
        Local_Branch_model = CovidAidAttend(combine_pneumonia)
#         Crop_model = CovidAidAttend(combine_pneumonia)

        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES)
    

    
    
    
    if os.path.exists(CKPT_PATH_G):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_G)
        else:                
            checkpoint = torch.load(CKPT_PATH_G,map_location='cpu')
        Global_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Global_Branch_model checkpoint")

    if os.path.exists(CKPT_PATH_L):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_L)
            
        else:
            checkpoint = torch.load(CKPT_PATH_L,map_location='cpu')
        Local_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Local_Branch_model checkpoint")

    if os.path.exists(CKPT_PATH_F):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_F)
        
        else:
            checkpoint = torch.load(CKPT_PATH_F,map_location='cpu')
        Fusion_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Fusion_Branch_model checkpoint")
            
    cudnn.benchmark = True
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    
    test(Global_Branch_model, Local_Branch_model, Fusion_Branch_model,
         test_loader,len(test_dataset),'test',cm_path,roc_path,combine_pneumonia)
    
def tensor_imshow(inp, title=None,bbox=None,img_name=None,size=None,vis_dir=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    if bbox is None:
        plt.imshow(inp, **kwargs)
        if title is not None:
            plt.title(title)
    else:
        rr,cc = rectangle_perimeter(bbox[:2],bbox[2:],shape=inp.shape)
#         plt.figure(figsize=(10, 5))
#         plt.axis('off')

#         plt.imshow(inp,**kwargs)
#         if title is not None:
#             plt.title(title)
        save_path=os.path.join(vis_dir,'vis_%s'%img_name[0])
#         print(save_path)
        rescaled= np.uint8(inp*255)
        color=np.array([255,0,0])
        set_color(rescaled,(rr,cc),color)
#         print(size)
        new_size =(480,480)

        img = Image.fromarray(rescaled).convert('RGB')
        img = img.resize(new_size)

        
        img.save(save_path)
        
#         plt.savefig(save_path)    

def visualize(img_dir= None,CKPT_PATH_G=None, CKPT_PATH_L=None, 
              CKPT_PATH_F=None, vis_dir=None, combine_pneumonia=True, binary_eval=True):
    
    if combine_pneumonia:
        N_CLASSES = 3
        CLASSES = ["Normal", "Pneumonia", "Covid"]
    else:
        N_CLASSES= 2
        CLASSES=["SEVERE","MILD"]
    
    print('********************load data********************')
    path = '../TestSet_Gamma/'
    image_path = path + 'TestSet_Gamma/'
    
    # for col_name in metadata_df.columns: 
    #     print(col_name, metadata_df[col_name].count())

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    
    test_dataset = CXR_Dataset_Visualization(image_dir=image_path,
                                   transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   normalize,
                                   ]))
    
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                    shuffle=False, num_workers=8, pin_memory=True)
    print('********************load data succeed!********************')

    print('********************load model********************')
    # initialize and load the model
    if USE_GPU:
        Global_Branch_model = CovidAidAttend(combine_pneumonia).cuda()
        Local_Branch_model = CovidAidAttend(combine_pneumonia).cuda()
        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES).cuda()
    else:
        Global_Branch_model = CovidAidAttend(combine_pneumonia)
        Local_Branch_model = CovidAidAttend(combine_pneumonia)
        Fusion_Branch_model = Fusion_Branch(input_size=2048, output_size=N_CLASSES)
    
    if os.path.exists(CKPT_PATH_G):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_G)
        else:                
            checkpoint = torch.load(CKPT_PATH_G,map_location='cpu')
        Global_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Global_Branch_model checkpoint")

    if os.path.exists(CKPT_PATH_L):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_L)
            
        else:
            checkpoint = torch.load(CKPT_PATH_L,map_location='cpu')
        Local_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Local_Branch_model checkpoint")

    if os.path.exists(CKPT_PATH_F):
        if USE_GPU:
            checkpoint = torch.load(CKPT_PATH_F)
        
        else:
            checkpoint = torch.load(CKPT_PATH_F,map_location='cpu')
        Fusion_Branch_model.load_state_dict(checkpoint)
        print("=> loaded Fusion_Branch_model checkpoint")
            
    cudnn.benchmark = True
    print('******************** load model succeed!********************')

    Global_Branch_model.eval()
    Local_Branch_model.eval()
    Fusion_Branch_model.eval()
    
    pred_global = torch.FloatTensor()
    pred_local = torch.FloatTensor()
    pred_fusion = torch.FloatTensor()
    pred_names=[]
    print("Generating Attention Maps")
    
    for i, (image,name, size) in tqdm(enumerate(test_loader), total=len(test_loader)):
        
        pred_names+=name
        if USE_GPU:
            input_var = torch.autograd.Variable(image.cuda())
        else:
            input_var = torch.autograd.Variable(image)
            
        output_global, fm_global, pool_global = Global_Branch_model(input_var)

        try :
            patchs_var, boundingList= Attention_gen_patchs(image, fm_global,'visualize')
        except ValueError:
            patchs_var, boundingList= Attention_gen_patchs(image, fm_global)

        output_local, _, pool_local = Local_Branch_model(patchs_var)
           
        output_fusion = Fusion_Branch_model(pool_global.data, pool_local.data)
        
        pred_global = torch.cat((pred_global,output_global.data.cpu()),0)
        pred_local = torch.cat((pred_local,output_local.data.cpu()),0)
        pred_fusion = torch.cat((pred_fusion,output_fusion.data.cpu()),0)
       
        tensor_imshow(image.squeeze(0),bbox=boundingList[0],img_name=name,size=size,vis_dir=vis_dir)
        
    if binary_eval:
        pred_global = torch.from_numpy(pred_global.numpy())
        pred_local = torch.from_numpy(pred_local.numpy())
        pred_fusion = torch.from_numpy(pred_fusion.numpy())
    
    scoresG = []
    for p,  n in zip(pred_global.numpy(), pred_names):
        p = ["%.1f %%" % (i * 100) for i in p]
#         l = np.argmax(l)
        scoresG.append([n] + p)
    scoresL = []
    for p,  n in zip(pred_local.numpy(), pred_names):
        p = ["%.1f %%" % (i * 100) for i in p]
#         l = np.argmax(l)
        scoresL.append([n] + p)
    scoresF = []
    for p,  n in zip(pred_fusion.numpy(), pred_names):
        p = ["%.1f %%" % (i * 100) for i in p]
#         l = np.argmax(l)
        scoresF.append([n] + p)
    
    header=['Name', 'Normal', 'Bacterial', 'Viral', 'COVID-19']
    alignment="c"*5
    if combine_pneumonia:
        header = ['Name', 'Normal', 'Pneumonia', 'COVID-19']
        alignment = "c"*4
    if binary_eval:
        header=["Name", "Non-Covid", "Covid"]
    
    predsFile=os.path.join(vis_dir,'predsGlobal.txt')
    f=open(predsFile,"w")
    pprint(header,f)
    pprint(scoresG,f)
    f.close()
    
    predsFile=os.path.join(vis_dir,'predsLocal.txt')
    f=open(predsFile,"w")
    pprint(header,f)
    pprint(scoresL,f)
    f.close()
    
    predsFile=os.path.join(vis_dir,'predsFusion.txt')
    f=open(predsFile,"w")
    pprint(header,f)
    pprint(scoresF,f)
    f.close()
    
    
    print("Visualizations Generated at: %s" %vis_dir)

    


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['train', 'test','visualize'], required=True)
    parser.add_argument("--combine_metadata", action='store_true', default=False)
    parser.add_argument("--save", type=str, default='pretrained_model/')
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--epochs", type=int,default=50)
    parser.add_argument("--resume",action='store_true',default=False)
    parser.add_argument("--ckpt_init",type=str,default='data/CovidXNet_transfered_3.pth.tar')
    parser.add_argument("--ckpt_G",type=str,default='./pretrained_model/AGCNN_Local_epoch_33.pth')
    parser.add_argument("--ckpt_L",type=str,default='./pretrained_model/AGCNN_Global_epoch_42.pth')
    parser.add_argument("--ckpt_F",type=str,default='./pretrained_model/AGCNN_Fusion_epoch_44.pth')
    parser.add_argument("--cm_path", type=str, default='plots/cm')
    parser.add_argument("--roc_path", type=str, default='plots/roc')
    parser.add_argument("--img_dir", type=str)
    parser.add_argument("--visualize_dir", type=str, default='./visualization/')
    parser.add_argument("--freeze", action='store_true', default=False)

    args = parser.parse_args()
        
    if args.mode=='train':
        if args.resume:
            train('init', args.ckpt_G, args.ckpt_L, args.ckpt_F, args.epochs,args.bs,True, 
                  args.save, args.combine_metadata, args.freeze)
        else:
            train(args.ckpt_init,'','','',args.epochs,args.bs,True,args.save,
                  args.combine_metadata, args.freeze)

    elif args.mode=='test':
        evaluate('TEST_LIST', args.ckpt_G, args.ckpt_L, args.ckpt_F, args.cm_path, args.roc_path, args.bs,
                 args.combine_metadata)
    
    else:
        visualize(args.img_dir, args.ckpt_G, args.ckpt_L, args.ckpt_F, args.visualize_dir, args.combine_metadata)





    


