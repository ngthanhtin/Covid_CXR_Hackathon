from pyexpat import model
import torch
import os

class config():
    model_name = 'resnet50' # 'resnet50'
    
    #data path
    path = '../TrainSet/'
    metadata_path = path + 'trainClinData.xls'
    #log path
    logpath = 'logs/'
    path_model_pretrained = './pretrained_models/{}'.format(model_name)

    #resume
    resume_model = ''
    
    # CLASS
    N_CLASSES= 2
    CLASSES=["SEVERE","MILD"]

    # train
    start_epoch = 0
    train_number_epochs = 20
    continue_train = False
    batch_size = 16
    optimizer = 'adam'
    scheduler = 'steprl'
    input_size = 256
    worker = 2
    num_class = 2
    fold = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # optimize
    LR = 1e-4
    
    lr_decay_step = 30
    lr_decay_gamma =  0.25
    weight_decay = 1e-5