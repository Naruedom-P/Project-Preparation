import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('20THB-17th-Banknote-Back.jpg')
img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

lower_color = np.array([29, 86, 6])
upper_coler = np.array([64, 255, 255])

mask = cv2.inRange(img_hsv, lower_color, upper_coler)

capture = cv2.VideoCapture(0)
ret, frame = capture.read()

frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_haight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

i=0
while(True):
    #ret, frame = capture.read()
    #gray = cv2.cvtColor(frame,cv2.COLOR_RGB2BGRA)

    has_frame, frame = capture.read()
    i = i+1

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, lower_color, upper_coler)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    n = len(contours)

    for i in range(0, n):
        cnt  = contours[i]
        perimeter = cv2.arcLength(cnt, True)
        if perimeter>500:
           x,y,w,h = cv2.boundingRect(cnt)
           cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),10)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
