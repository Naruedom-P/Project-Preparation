import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
img1 = cv2.imread('20THB-17th-Banknote-Back.jpg',0)
img2 = cv2.imread('20.2.jpg',0)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)



img=cv2.drawKeypoints(img2,kp2,img2)
cv2.imwrite('255.jpg',img)
