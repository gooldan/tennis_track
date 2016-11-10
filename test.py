import cv2
import numpy as np

back = cv2.imread('frame21.jpg', 0)
cap = cv2.VideoCapture('Main2.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC,470000)

while True:
    

    _, frame = cap.read()
    
    mask = np.zeros(frame.shape, dtype=np.uint8)
    roi_corners = np.array([[(0,720),(0,460),(440,280),(880,280),(1280,460),(1280,720)]], dtype=np.int32)
    channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your img
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    
    frame = cv2.bitwise_and(frame, mask)
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.subtract(frame, back)
    
    _, thresh = cv2.threshold(diff, 10, 255, 0)
    
    diff = cv2.bitwise_and(diff, diff, mask = thresh)
    
#    _, _, _ = cv2.findContours(diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('diff', diff)
    cv2.imshow('thresh', thresh)
    
    if cv2.waitKey(25) & 0xFF == 27:
        break

cv2.destroyAllWindows()