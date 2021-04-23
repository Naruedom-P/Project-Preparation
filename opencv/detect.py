import cv2
import numpy as np
import matplotlib.pyplot as plt



rin = cv2.imread('20THB-17th-Banknote-Back.jpg')
rin_hsv = cv2.cvtColor(rin,cv2.COLOR_BGR2HSV)
inr = cv2.inRange(rin_hsv,np.array([20,50,0]),np.array([50,255,255]))
mask = cv2.cvtColor(inr,cv2.COLOR_GRAY2BGR)!=0
cv2.imwrite('t1.jpg',rin*mask)


