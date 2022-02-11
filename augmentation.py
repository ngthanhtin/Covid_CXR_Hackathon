import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import config
import copy
import matplotlib.pyplot as plt

# transforms.RandomAffine(30, scale=(0.8,1.2), shear=[-15,15,-15,15]),

def get_augmentation(phase):
    if phase == "train":
        return  A.Compose([
                    A.Resize(height=config.input_size, width=config.input_size),
                    A.CenterCrop(height=256, width=256),
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
                    A.Cutout(max_h_size=int(config.input_size * 0.1), max_w_size=int(config.input_size * 0.1), num_holes=5, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
        # return A.Compose([
        #                 A.Transpose(p=0.5),
        #                 A.VerticalFlip(p=0.5),
        #                 A.HorizontalFlip(p=0.5),
        #                 A.RandomBrightness(limit=0.2, p=0.75),
        #                 A.RandomContrast(limit=0.2, p=0.75),
        #                 A.OneOf([
        #                     A.MotionBlur(blur_limit=5),
        #                     A.MedianBlur(blur_limit=5),
        #                     A.GaussianBlur(blur_limit=5),
        #                     A.GaussNoise(var_limit=(5.0, 30.0)),
        #                 ], p=0.7),

        #                 A.OneOf([
        #                     A.OpticalDistortion(distort_limit=1.0),
        #                     A.GridDistortion(num_steps=5, distort_limit=1.),
        #                     A.ElasticTransform(alpha=3),
        #                 ], p=0.7),

        #                 A.CLAHE(clip_limit=4.0, p=0.7),
        #                 A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        #                 A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        #                 A.Resize(config.input_size, config.input_size),
        #                 A.CenterCrop(height=256, width=256),
        #                 A.Cutout(max_h_size=int(config.input_size * 0.375), max_w_size=int(config.input_size * 0.375), num_holes=1, p=0.7),    
        #                 A.Normalize(),
        #                 ToTensorV2()
        #             ])
    elif phase in ['test','valid']:
        return A.Compose([
            A.Resize(height=config.input_size, width=config.input_size),
            A.CenterCrop(height=256, width=256),
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