import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from config import config

class CXRClassifier(nn.Module):
    def __init__(self, n_labels):
        super(CXRClassifier, self).__init__()

        if config.model_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=True)
            in_feature = self.backbone.classifier.in_features
            self.backbone.classifier = torch.nn.Sequential(
                    torch.nn.Linear(in_feature, n_labels))

        elif config.model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
            self.backbone = models.resnext50_32x4d(pretrained=True)
            in_feature = self.backbone.fc.in_features
            self.backbone.fc = torch.nn.Sequential(
                    torch.nn.Linear(in_feature, n_labels))

        elif config.model_name == 'swin224':
            # https://github.com/rwightman/pytorch-image-models/blob/ef72ad417709b5ba6404d85d3adafd830d507b2a/timm/models/swin_transformer.py#L47-L89
            model_architecture = "swin_large_patch4_window7_224" # pretrained with classifier output with 1000 classes
            self.backbone = timm.create_model(model_architecture, pretrained=True)
            num_input_features = self.backbone.head.in_features # pretrained model's default fully connected Linear Layer
            self.backbone.head = nn.Linear(in_features=num_input_features, out_features=1024, bias=True)  # replacing output with class number
            self.cls_head = nn.Linear(1024, n_labels, bias=True)
    
    def forward(self, x):
        x = self.backbone.features(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # x = self.cls_head(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        
        return x

    

