import cv2 
import numpy as np
import matplotlib.pyplot as plt

teto = cv2.imread('IMG_1.jpg') # อ่านภาพสี
teto_gr = cv2.cvtColor(teto,cv2.COLOR_BGR2GRAY) # แปลงเป็นขาวดำ
_,teto_thr = cv2.threshold(teto_gr,10,255,0)
# แปลงสัณฐานเพื่อลดจุดปนเปื้อน
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
teto_thr = cv2.morphologyEx(teto_thr,cv2.MORPH_CLOSE,kernel)
# หาเค้าโครง
contour,_ = cv2.findContours(teto_thr,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
n_contour = len(contour)
# สีของเส้นเค้าโครงแต่ละเส้น
si = plt.get_cmap('rainbow')(np.arange(n_contour)/(n_contour-1))[:,[2,1,0]]*255
# สร้างเค้าโครงใหม่ที่ย่อลงจากเดิมเล็กน้อย
contour_dp1 = [cv2.approxPolyDP(cnt,4.5,True) for cnt in contour]
# สร้างเค้าโครงใหม่ที่ย่อลงจากเดิมไปค่อนข้างมาก
contour_dp2 = [cv2.approxPolyDP(cnt,11,True) for cnt in contour]
teto_cnt = teto.copy()
teto_cnt_dp1 = teto.copy()
teto_cnt_dp2 = teto.copy()
for i in range(n_contour):
    teto_cnt = cv2.drawContours(teto_cnt,contour,i,si[i],2)
    teto_cnt_dp1 = cv2.drawContours(teto_cnt_dp1,contour_dp1,i,si[i],2)
    teto_cnt_dp2 = cv2.drawContours(teto_cnt_dp2,contour_dp2,i,si[i],2)

cv2.imwrite('teto14c02.jpg',teto_cnt)
cv2.imwrite('teto14c03.jpg',teto_cnt_dp1)
cv2.imwrite('teto14c04.jpg',teto_cnt_dp2)
