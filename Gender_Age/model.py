# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
import torch.nn as nn
import torchvision

class AgeModel(nn.Module):
    """Model recognition gender and age base on EfficientNetB0 model"""
    def __init__(self):
        super().__init__()

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        efnet = torchvision.models.efficientnet_b0(weights=weights)
        # Freeze some layers in the "features" section of the model by setting requires_grad=False
        for i, param in enumerate(efnet.features.parameters()):
            if i < 180:
                param.requires_grad = False

        # Recreate the classifier layer
        self.efnet = efnet
        self.efnet.classifier = nn.Linear(1280, 512)

        # Branch of gender
        self.gender_ = nn.Sequential(nn.Dropout(0.2),
                                     nn.Linear(512, 1))

        # Branch of age
        self.age_ = nn.Sequential(nn.Dropout(0.2),
                                  nn.Linear(512, 1),
                                  nn.ReLU())

    def forward(self, x):
        x = self.efnet(x)

        # Split 2 branch
        age = self.age_(x)
        gender = self.gender_(x)

        return gender, age
