# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import os
import torch
import config
import math
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class AgeDataset(Dataset):
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.transform = transform
        self.image_paths = os.listdir(self.data_root)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.image_paths[index])
        img = np.array(Image.open(img_path).convert('RGB'))
        augmentations = self.transform(image=img)
        img = augmentations['image']
        age = self.image_paths[index][:3]
        if age[1] == '_':
            age = age[0]
        elif age[2] == '_':
            age = age[:2]
        age = int(age)
        if age < 10:
            gender = self.image_paths[index][2]
        elif age > 99:
            gender = self.image_paths[index][4]
        else:
            gender = self.image_paths[index][3]
        gender = float(gender)
        return img, gender, age / 100

def Data_Loader(data_root,
                batch_size: int,
                num_workers: int,
                train_size=0.9,
                ):

    data = AgeDataset(data_root, config.transform)
    train = len(data)*train_size
    train = math.ceil(train)
    val = len(data) - train
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train, val])

    # Turn images into data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return train_dataloader, val_dataloader
