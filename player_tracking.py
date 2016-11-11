import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('Main2.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC,470000)
back = cv2.imread('back.jpg', 0)
back2 = cv2.imread('frame21.jpg', 0)

while True:
    ret, frame = cap.read()
    
    if ret == True:
        
        # setting a ROI
        mask = np.zeros(frame.shape, dtype=np.uint8)
        roi_corners = np.array([[(0,720),(0,460),(440,280),(880,280),(1280,460),(1280,720)]], dtype=np.int32)
        channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your img
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        frame = cv2.bitwise_and(frame, mask)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # background subtraction
        diff = cv2.subtract(frame, back2)
        
        _, thresh = cv2.threshold(diff,10,255, 0)
        
        # contours search
        _, cont, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # searching for the maximal area contour
        # I am sure that it is player's body
        maxArea = 0
        maxCnt = []
        for elem in cont:
            area = cv2.contourArea(elem)
            if(area>maxArea):
                maxArea = area
                cnt = elem
        maxCnt.append(cnt)

        # adding such elements as rocket and feet to contours array
        manCnt = []
        for elem in cont:
            if cv2.contourArea(elem) > maxArea * 0.03:
                manCnt.append(elem)
        cv2.drawContours(thresh, manCnt, -1, 200, -1)
        
        # polygonal approximation of player's contour
        approxCnt = []        
        for elem in manCnt:
            epsilon = 0.01*cv2.arcLength(elem,True)
            approx = cv2.approxPolyDP(elem,epsilon,True)
            approxCnt.append(approx) 
        cv2.drawContours(thresh, approxCnt, -1, 50, -1)
        
        cv2.imshow('diff', diff)
        cv2.imshow('thresh', thresh)
  
        if cv2.waitKey(25) & 0xFF == 27:
            break
        
    else:
        print('ACHTUNG!')
        break

cap.release()
cv2.destroyAllWindows()