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
    """Create child class of Dataset.

    Initialize class to take the input path and return
    image, gender and age.

    Arg:
        data_root: Root of image file.
        transform: A set of transform to convert input image.

    Return:
         img: Image after covert.
         gender: Gender of human.
         age(normalize): Age of human.
    """
    def __init__(self, data_root, transform):
        self.data_root = data_root
        self.transform = transform
        self.image_paths = os.listdir(self.data_root)

    # Take len of paths
    def __len__(self):
        return len(self.image_paths)

    # Take item
    def __getitem__(self, index):
        # Take input path
        img_path = os.path.join(self.data_root, self.image_paths[index])
        img = np.array(Image.open(img_path).convert('RGB'))

        # Data augmentation
        augmentations = self.transform(image=img)
        img = augmentations['image']

        # Take the age
        age = self.image_paths[index][:3]
        if age[1] == '_':
            age = age[0]
        elif age[2] == '_':
            age = age[:2]
        age = int(age)

        # Take the gender
        if age < 10:
            gender = self.image_paths[index][2]
        elif age > 99:
            gender = self.image_paths[index][4]
        else:
            gender = self.image_paths[index][3]
        gender = float(gender)

        return img, gender, age / 116

def Data_Loader(data_root,
                batch_size: int,
                num_workers: int=2,
                train_size=0.9,
                ):
    """Function for fast setup.
    Take some input parameters and return DataLoader.

    :param data_root: Root of data file.
    :param batch_size: Number of batch size.
    :param num_workers: Number of worker (Default: 2)
    :param train_size: Size of train data (Default: 0.9)
    :return:
    """

    # Take dataset
    data = AgeDataset(data_root, config.transform)

    # Convert size
    train = len(data)*train_size
    train = math.ceil(train)
    val = len(data) - train
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train, val])

    # Turn dataset into data loaders
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
