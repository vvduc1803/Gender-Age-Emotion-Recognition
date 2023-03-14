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
LR = 1e-3
WEIGHT_DECAY = 1e-5
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "Gender_Age.pt"

"""Gender"""
class_names = ['Male', 'Female']

"""Transform for dataset"""
transform = A.Compose(
    [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
     A.HorizontalFlip(p=0.5),
     A.ToGray(True),
     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
     ToTensorV2(),
     ],
)

