import sys
sys.path.insert(0, './')
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import cv2
import numpy as np
import torch
from torchvision import transforms
from models import CXRClassifier
from config import config

def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

 #load model
model_path = config.path_model_pretrained + '_best_073_epoch19.pt'

model = CXRClassifier(n_labels=config.N_CLASSES)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

target_layers = [model.backbone.features.denseblock4]
image_path = "../TestSet/TestSet/P_1_108.png"
img = cv2.imread(image_path)
# pre-process the image
img = cv2.resize(img, (256,256))
img = np.float32(img) / 255
# Opencv loads as BGR:
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_tensor = preprocess_image(img)

# Construct the CAM object once, and then re-use it on many images:
cam = ScoreCAM(model=model.backbone, target_layers=target_layers, use_cuda=True)

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.
targets = None
# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# In this example grayscale_cam has only one image in the batch:
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
cv2.imshow("cam", visualization)
cv2.waitKey(0)