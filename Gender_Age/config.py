# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""All task can tuning"""
DATASET = 'UTKFace/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKER = 4
IMAGE_SIZE = 200
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
WEIGHT_DECAY = 1e-2
PIN_MEMORY = True
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = "Gender_Age.pt"

"""Gender"""
class_names = ['Male', 'Female']

"""Transform for dataset"""
transform = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
     A.HorizontalFlip(p=0.5),
     A.ColorJitter(p=0.2),
     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0, ),
     ToTensorV2(),
     ],
)

