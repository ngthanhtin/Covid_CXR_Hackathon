import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm

def _find_index(ds, desired_label):
    desired_index = None
    for ilabel, label in enumerate(ds.labels):
        if label.lower() == desired_label:
            desired_index = ilabel
            break
    if not desired_index is None:
        return desired_index
    else:
        raise ValueError("Label {:s} not found.".format(desired_label))

class CXRClassifier(nn.Module):
    def __init__(self, n_labels):
        super(CXRClassifier, self).__init__()

        self.backbone = models.densenet121(pretrained=True)
        # self.backbone = models.resnet50(pretrained=True)
        in_feature = self.backbone.classifier.in_features
        # in_feature = self.backbone.fc.in_features
        self.backbone.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_feature, n_labels))
        # self.backbone.fc = torch.nn.Sequential(
        #         torch.nn.Linear(in_feature, n_labels))

    def forward(self, x):
        x = self.backbone(x)
        return x

    

