# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:39:35 2016

@author: HP
"""

import cv2
import numpy as np

cap = cv2.VideoCapture('Main2.mp4')
isPaused = False

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.putText(img,str(x)+' '+str(y), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        global isPaused
        isPaused = not isPaused

# Create a black img, a window and bind the function to window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('img')
cv2.setMouseCallback('img',draw_circle)







#
#import cv2
#import numpy as np

## original img
## -1 loads as-is so if it will be 3 or 4 channel as the original
#img = cv2.imread('img.png', -1)
## mask defaulting to black for 3-channel and transparent for 4-channel
## (of course replace corners with yours)
#mask = np.zeros(img.shape, dtype=np.uint8)
#roi_corners = np.array([[(10,10), (300,300), (10,300)]], dtype=np.int32)
## fill the ROI so it doesn't get wiped out when the mask is applied
#channel_count = img.shape[2]  # i.e. 3 or 4 depending on your img
#ignore_mask_color = (255,)*channel_count
#cv2.fillPoly(mask, roi_corners, ignore_mask_color)
#
## apply the mask
#masked_img = cv2.bitwise_and(img, mask)
#
## save the result
#cv2.imwrite('img_masked.png', masked_img)









while(1):
    if not isPaused:
        ret, img = cap.read()
#        mask = np.zeros(img.shape, dtype=np.uint8)
#        roi_corners = np.array([[(0,360),(0,230),(234,100),(400,100),(640,230),(640,360)]], dtype=np.int32)
#        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your img
#        ignore_mask_color = (255,)*channel_count
#        cv2.fillPoly(mask, roi_corners, ignore_mask_color)

        # apply the mask
#        img = cv2.bitwise_and(img, mask)
        #cv2.imshow('img', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()