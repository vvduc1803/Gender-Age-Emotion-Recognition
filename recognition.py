# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
import torch
import cv2
import mediapipe as mp
import config
from Emotion.model import EmotionModel
from Gender_Age.model import AgeModel
from utils import load_checkpoint

class FaceDetector:
    """
    Find faces in realtime using the light weight model provided in the mediapipe
    library.
    """

    def __init__(self, minDetectionCon=0.5):
        """
        :param minDetectionCon: Minimum Detection Confidence Threshold
        """

        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionCon)

    def findFaces(self, img, draw=True):
        """
        Find faces in an image and return the bbox info
        :param img: Image to find the faces in.
        :param draw: Flag to draw the output on the image.
        :return: Bounding Box list. [x_min, y_min, w, h]

        """

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)

                bboxInfo = {"id": id, "bbox": bbox}
                bboxs.append(bboxInfo)
                if draw:
                    img = cv2.rectangle(img, bbox, (255, 0, 255), 2)

                    cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                                (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        return img, bboxs

    def Load_model(self):
        """Load 2 model are Gender_Age Model and Emotion Model"""
        # Initialize 2 model
        age_model = AgeModel().to(config.device)
        emotion_model = EmotionModel().to(config.device)

        # Load weight of 2 model
        load_checkpoint(config.Gender_Age_Model_Path, age_model)
        load_checkpoint(config.Emotion_Model_Path, emotion_model)

        return age_model, emotion_model

    def Recognition(self, image, age_model, emotion_model):
        """Recognition people in image"""
        # Apply transform
        image = config.age_transfrom(image).to(config.device)
        image = torch.unsqueeze(image, dim=0)

        # Recognition gender and age
        gender_score, age = age_model(image)
        gender_score = torch.sigmoid(gender_score)
        gender = 1 if gender_score > 0.5 else 0
        print(gender_score)
        gender_score = (1-gender_score) if gender_score < 0.5 else (gender_score-0.5)

        # Apply transform
        image = config.emotion_transfrom(image)

        # Recognition emotion
        emotion = emotion_model(image)
        score_emotion = torch.max(torch.softmax(emotion, dim=1))
        emotion = torch.argmax(torch.softmax(emotion, dim=1), dim=1)

        return gender, gender_score.item(), age, emotion, score_emotion.item()

