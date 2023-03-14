# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary package"""
import torch
import torchvision.transforms as transforms
import Emotion.config as emotion_conf
import Gender_Age.config as gender_age_conf

"""All task can tuning"""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
AGE_SIZE = 200
EMOTION_SIZE = 380
Emotion_Model_Path = 'Emotion/Emotion.pt'
Gender_Age_Model_Path = 'Gender_Age/Gender_Age.pt'

gender_names = gender_age_conf.class_names
emotion_names = emotion_conf.classes_names

emotion_lr = emotion_conf.LR
gender_age_lr = gender_age_conf.LR

age_transfrom = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Grayscale(3),
     transforms.Resize(AGE_SIZE),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])

emotion_transfrom = transforms.Compose(
    [transforms.Resize(EMOTION_SIZE),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])