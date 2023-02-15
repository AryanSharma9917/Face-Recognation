import cv2
import numpy as np

CLASSIFIER_PATH = "C:\Users\kumar\OneDrive\GitHub\Face_Recognition\Classifier\haarcascade_frontalface_default.xml"
classifier = cv2.CascadeClassifier(CLASSIFIER_PATH)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainData.yml')

cam = cv2.VideoCapture(0)
id = 0

while(True):
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)
    name = ''
    for face in faces:
        (x, y, w, h) = face
        cv2.rectangle(frame, (x, y), (x+h, y+h), (0, 255, 255), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if(conf < 50):
            if(id == 1):
                name = "Proyanka Chopra"
                frame = cv2.putText(frame, name, (x, y+h),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 3)
    cv2.imshow('Face', frame)

    if(cv2.waitKey(1) == 27):
        break

cam.release()
cv2.destroyAllWindows
