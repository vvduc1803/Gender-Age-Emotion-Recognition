# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""
An implementation of EfficientNet CNN architecture.
"""
import torch
import torchvision

def EmotionModel():
    """Model build base on pretrained EfficientNetB4"""

    weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
    model = torchvision.models.efficientnet_b4(weights=weights)

    # Freeze some base layers in the "features" section of the model by setting requires_grad=False
    for i, param in enumerate(model.features.parameters()):
        if i < 400:
            param.requires_grad = False

    # Recreate the classifier layer
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(1792, 512),
        torch.nn.SiLU(),
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features=512,
                        out_features=7,  # same number of output units as our number of classes
                        bias=True))
    return model


