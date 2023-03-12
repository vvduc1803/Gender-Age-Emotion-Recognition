# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
import torch
import torch.nn as nn
import torchvision
from torchinfo import summary

# Model regconition gender and age
class AgeModel(nn.Module):
    def __init__(self, inchan=3, hidden=32):
        super().__init__()

        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        efnet = torchvision.models.efficientnet_b0(weights=weights)



        # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
        for param in efnet.features.parameters():
            param.requires_grad = False

        # Recreate the classifier layer and seed it to the target device
        self.efnet = efnet
        self.efnet.classifier = nn.Linear(1280, 512)

        self.gender_ = nn.Sequential(nn.Dropout(0.2),
                                     nn.Linear(512, 1))

        self.age_ = nn.Sequential(nn.Dropout(0.2),
                                  nn.Linear(512, 1),
                                  nn.ReLU())

    def forward(self, x):
        x = self.efnet(x)

        # Split 2 branch
        age = self.age_(x)
        gender = self.gender_(x)

        return gender, age

# def test():
#     model = AgeModel()
#     x = torch.rand((1, 3, 224, 224))
#     y = model(x)
#     print(y)
#     summary(model, (1, 3, 224, 224))
# test()
