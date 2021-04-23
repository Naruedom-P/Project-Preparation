import cv2
import numpy as np
import matplotlib.pyplot as plt

frame = cv2.imread('abba.png')


img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()
    print(ret)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(img_gray, 1.3, 4)

  
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

  
    plt.axis("off")
    plt.imshow('Video',  frame)
    plt.show()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
