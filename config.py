import torch
import os

class config():
    
    path_model_pretrained = './pretrained_model/densenet121.pth'
    path = './TrainSet/'
    metadata_path = path + 'trainClinData.xls'
    # path save file
    model_name = 'densenet121' # densenet121, resnet50
    optimizer = 'adam'
    resume_model = ''
    
    # train
    start_epoch = 0
    train_number_epochs = 50
    continue_train = False
    batch_size = 32
    num_embeddings = 512
    input_size = 224
    worker = 2
    num_class = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # optimize
    LR_G = 1e-3
    LR_L = 1e-3
    LR_F = 1e-4
    
    lr_decay_step = 30
    lr_decay_gamma =  0.25
    weight_decay = 1e-4