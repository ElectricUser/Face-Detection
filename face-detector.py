import cv2 as cv
import numpy as np

FACE_CASCADE = cv.CascadeClassifier('models/haarcascade_frontalface_default.xml')


def face_detection(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, minNeighbors=4)
    img_cp = np.copy(frame)
    for (x, y, w, h) in faces:
        cv.rectangle(img_cp, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_cp


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    face = face_detection(frame)
    cv.imshow('frame', face)

    if cv.waitKey(1) == ord('q'):
        break


cv.destroyAllWindows()
