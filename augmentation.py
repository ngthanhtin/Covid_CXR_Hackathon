import torchvision
import torchvision.transforms as transforms
from config import config
def get_augmentation(phase):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    if phase == "train":
        return transforms.Compose([
                            transforms.Resize(config.input_size),
                            transforms.RandomAffine(30, scale=(0.8,1.2), shear=[-15,15,-15,15]),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                            ])
        
    elif phase in ['test','valid']:
        return transforms.Compose([
                            transforms.Resize(config.input_size),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                                    ])
    
    