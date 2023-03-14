# -*- coding: utf-8 -*-
"""
@author: Van Duc <vvduc03@gmail.com>
"""
"""Import necessary packages"""
import cv2
import config

from recognition import FaceDetector

# Load video
cap = cv2.VideoCapture('c.mp4')

# Load model
detector = FaceDetector()
gender_age_model, emotion_model = detector.Load_model()

while cap.isOpened():
    success, frame = cap.read()

    # Use mediapipe library detect face
    _, bbox = detector.findFaces(frame, draw=False)

    if bbox:
        # Load each person in frame
        for box in bbox:

            # Extract coordinates
            x_min, y_min, w, h = box["bbox"]

            # Avoid some problem like: person zoom close camera, person out camera
            try:
                # Crop frame for recognition
                face = frame[y_min-40:y_min+h, x_min-20:x_min+w+20]

                # Show crop image
                # cv2.imshow('Face', face)

                # Make recognition
                idx_gender, gender_score, age, idx_emotion, emotion_score = detector.Recognition(face,
                                                                                                 gender_age_model,
                                                                                                 emotion_model)

                # Put information into screen
                cv2.rectangle(frame, (x_min-20, y_min-40), (x_min+w+20, y_min+h), (0, 0, 0), 2)
                cv2.rectangle(frame, (x_min+w+20, y_min-41), (x_min+w+200, y_min+55), (0, 0, 0), -1)

                cv2.putText(frame,
                            f'Gender: {config.gender_names[int(idx_gender)]}-{gender_score*100:.0f}%',
                            (x_min+w+20, y_min-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame,
                            f'Age: {age.item() * 116 - 5:.0f}-{age.item() * 116 + 5:.0f}',
                            (x_min+w+20, y_min+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f'Emotion: {config.emotion_names[int(idx_emotion)]}-{emotion_score*100:.0f}%',
                            (x_min+w+20, y_min+50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            except cv2.error:
                None
            except RuntimeError:
                None

    # Show result
    cv2.imshow('MediaPipe Face Detection', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()