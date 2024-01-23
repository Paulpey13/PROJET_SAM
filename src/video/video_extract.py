import cv2
import numpy as np
import pandas as pd
from keras.models import Model
import matplotlib.pyplot as plt

# Fonction pour extraire des segments de la vidéo
def extract_video_segments(video_file_path, start_ms, end_ms):
    cap = cv2.VideoCapture(video_file_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_ms * fps / 1000)
    end_frame = int(end_ms * fps / 1000)
    
    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or current_frame > end_frame:
            break
        if current_frame >= start_frame:
            frames.append(frame)
        current_frame += 1
    cap.release()
    return frames

# Fonction de prétraitement de chaque cadre
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (224, 224))  # Redimensionnement pour VGG16
    return frame_resized

# Classe pour l'extraction de caractéristiques
class FeatureExtractor:
    def __init__(self, min_num_of_points=68):
        modelFile = "model_video/res10_300x300_ssd_iter_140000.caffemodel"
        configFile = "model_video/deploy.prototxt"
        self.face_net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        self.landmark_model = cv2.face.createFacemarkLBF()
        model_path = 'model_video/lbfmodel.yaml'
        self.landmark_model.loadModel(model_path)
        self.expected_num_of_points = min_num_of_points

    def extract(self, frame):
        h, w = frame.shape[:2]  # Définir la hauteur et la largeur de l'image
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        all_keypoints = []

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_array = np.array([[startX, startY, endX-startX, endY-startY]], dtype=np.int32)
                success, landmarks = self.landmark_model.fit(gray, faces_array)
                if success and landmarks is not None:
                    for landmark_set in landmarks:
                        for landmark in landmark_set:
                            for point in landmark:
                                x, y = int(point[0]), int(point[1])
                                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                        all_keypoints.extend(landmark_set)

        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.show()
        return all_keypoints

# Fonction pour fusionner les caractéristiques
def fuse_features(features):
    # Vous devrez décider de la logique de fusion selon vos besoins
    return features if features else [np.zeros((68, 2))]  # Exemple: retourne une liste de points zéros si vide
