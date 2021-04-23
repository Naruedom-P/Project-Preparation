import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
#print(MIN_MATCH_COUNT)

img1 = cv2.imread('20.2.jpg',0)
img2 = cv2.imread('20THB-17th-Banknote-Back.jpg',0)

cv2.imshow('img1',img1)
cv2.imshow('img2',img2)

cap = cv2.VideoCapture('VID_20.mp4')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    sift = cv2.SIFT()

    # ค้นหาประเด็นสำคัญและตัวอธิบายด้วย SIFT
    #kp1, des1 = sift.detectAndCompute(img1,None)
    #kp2, des2 = sift.detectAndCompute(img2,None)
    kp1 = sift.detect(img1,None)
    kp2 = sift.detect(frame,None)
    
    des1 = sift.compute(img1,kp1)
    des2 = sift.compute(frame,kp2)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
            
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    plt.imshow(img3, 'gray'),plt.show()


    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
