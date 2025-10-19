import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_tfms(img_size=320):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=4),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2(),
    ])

def get_test_tfms(img_size=320):
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=4),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2(),
    ])
