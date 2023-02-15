import cv2   
import numpy as np

CLASSIFIER_PATH = "C:/Users/kumar/OneDrive/GitHub/Face_Recognition/Classifier/haarcascade_frontalface_default.xml"
classifier = cv2.CascadeClassifier(CLASSIFIER_PATH)

cam = cv2.VideoCapture(0)

id = input('ID: ')
num = 0

while(True):
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 5)

    for face in faces:
        num += 1
        (x, y, w, h) = face
        cv2.imwrite('C:/Users/kumar/OneDrive/GitHub/Face_Recognition/dataset/User.'+str(id)+"."+str(num)+".jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(1)

    cv2.imshow('Frame ', frame)
    cv2.waitKey(1)

    if(num > 60):
        break

cam.release()
cv2.destroyAllWindows