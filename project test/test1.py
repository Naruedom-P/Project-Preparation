import cv2
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import cv2 as cv
faceCascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


from skimage import io

img = io.imread('IMG_1.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

io.imshow(img)
io.show()


faces = faceCascade.detectMultiScale(gray, 1.1, 3)

print("number of face(s)", faces.shape[0])

tmp_img = np.copy(img)

for (x,y,w,h) in faces:
  io.imshow(tmp_img[y:y+h, x:x+w])
  io.show()

  
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
    plt.imshow('Video',  gr_image)
    plt.show()


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
