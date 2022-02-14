import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import config
import copy
import matplotlib.pyplot as plt

def get_augmentation(phase):
    if phase == "train":
        return  A.Compose([
                    A.Resize(height=config.input_size, width=config.input_size),
                    A.CenterCrop(height=224, width=224),
                    A.ToGray(p=0.01),
                    A.OneOf([
                       A.GaussNoise(var_limit=[10, 50]),
                       A.GaussianBlur(),
                       A.MotionBlur(),
                       A.MedianBlur(),
                      ], p=0.2),
                    A.OneOf([
                       A.OpticalDistortion(distort_limit=1.0),
                       A.GridDistortion(num_steps=5, distort_limit=1.),
                       A.ElasticTransform(alpha=3),
                   ], p=0.2),
                     A.OneOf([
                         A.CLAHE(),
                         A.RandomBrightnessContrast(),
                     ], p=0.25),
                     A.HueSaturationValue(p=0.25),
                    A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
                    # A.Cutout(max_h_size=int(config.input_size * 0.1), max_w_size=int(config.input_size * 0.1), num_holes=5, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
    elif phase in ['test','valid']:
        return A.Compose([
            A.Resize(height=config.input_size, width=config.input_size),
            A.CenterCrop(height=224, width=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # A.Normalize(),
            ToTensorV2()
        ])
    

def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i in range(samples):
        image, _ = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()