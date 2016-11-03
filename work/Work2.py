import cv2
import numpy as np

backg = cv2.imread('Background.png',0)
cap = cv2.VideoCapture("Tennis1.mp4")
cap.set(cv2.CAP_PROP_POS_MSEC,350000)
perFrame=False
while(1):

# we need to get a grayscale frame
 ret, frame = cap.read()

 img3 = cv2.subtract(frame, backg)
 ret,thresh = cv2.threshold(img3,10,255,0)
 cv2.imshow("thresh", thresh)
 cv2.waitKey()

 _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

 resCnt = []
 for elem in contours:
  area = cv2.contourArea(elem)
  if(area>100):
   rect = cv2.minAreaRect(elem)
   box = cv2.boxPoints(rect)
   box = np.int0(box)
   cv2.drawContours(thresh,[box],-1,(255,255,255),2)

 cv2.imshow("contours", thresh)

 k = cv2.waitKey(delay)
 if k == 27:
  break
cap.release()
cv2.destroyAllWindows()









