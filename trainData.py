import os
import cv2
import numpy as np
from PIL import Image

DATASET_PATH = 'C:\Users\kumar\OneDrive\GitHub\Face_Recognition\dataset'

recognizer = cv2.face.LBPHFaceRecognizer_create()


def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []

    for imagePath in imagePaths:
        face = Image.open(imagePath).convert('L')
        face_np = np.array(face)
        id = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(face_np)
        ids.append(id)
        cv2.imshow('Training', face_np)
        cv2.waitKey(10)
    return ids, faces


ids, faces = getImagesWithID(DATASET_PATH)

recognizer.train(faces, np.array(ids))
recognizer.save('trainData.yml')
cv2.destroyAllWindows()
