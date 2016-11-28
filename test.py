import cv2
import numpy as np
from matplotlib import pyplot as plt

cap = cv2.VideoCapture('Main2.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC, 450000)
back = cv2.imread('back.jpg', 0)
back2 = cv2.createBackgroundSubtractorMOG2(history=50, detectShadows=0)
ballfound = False
xprev=10000
yprev=10000

while True:
    ret, frame = cap.read()
    
    if ret == True:
        
        # setting a ROI
        mask = np.zeros(frame.shape, dtype=np.uint8)
        roi_corners = np.array([[(0,720),(0,460),(480,230),(800,230),(1280,460),(1280,720)]], dtype=np.int32)
        channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your img
        ignore_mask_color = (255,)*channel_count
        cv2.fillPoly(mask, roi_corners, ignore_mask_color)
        frame = cv2.bitwise_and(frame, mask)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # background subtraction block
#        diff = cv2.subtract(frame, back)
        diff = back2.apply(frame)
        _, diff = cv2.threshold(diff,5,255, cv2.THRESH_BINARY_INV)
        
        
        # doubtful part
        #kernel = np.ones((2,2),np.uint8)
        #diff = cv2.dilate(diff,kernel,iterations = 2)
        #diff = cv2.erode(diff,kernel,iterations = 2)
        
        cv2.imshow('diff', diff)
        
        params = cv2.SimpleBlobDetector_Params()

        params.filterByConvexity = True
        params.minConvexity = 0.8
    
        params.filterByInertia = True
        params.minInertiaRatio = 0.3
        
        params.filterByArea = True
        params.minArea = 15
        params.maxArea = 300
 
        params.filterByCircularity = True
        params.minCircularity = 0.3


        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(diff)
         
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

	# сделаем цветастенько
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        
        
        
#        # shit begins
#        
        # проверяем, видим ли мы мячик в окрестности его последнего расположения
        ballfound = False
        for elem in keypoints:
            if ((abs(elem.pt[0]-xprev)<50) or (abs(elem.pt[1]-yprev)<50)):
                ballfound = True
        
        
        # видим
	# находим самый близкий к предыдущей дислокации мячика блоб и нарекаем его мячиком
        xmin=9000
        ymin=9000
        if ballfound:
            for elem in keypoints:
                if ((abs(elem.pt[0]-xprev)<xmin)&(abs(elem.pt[1]-yprev)<ymin)):
                    xmin = abs(elem.pt[0]-xprev)
                    ymin = abs(elem.pt[1]-yprev)
                    xcur=elem.pt[0]
                    ycur=elem.pt[1]
            cv2.circle(frame, (int(xcur), int(ycur)), 5, (33, 255, 237), 2) 
            cv2.putText(frame, 'BALL', (int(xcur)+10, int(ycur)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (33, 255, 237), 1)
            xprev=xcur
            yprev=ycur
           
	# не видим
	# тут я не смог придумать, как найти площадь блоба по кейпоинту, поэтому мячом нарекается последний контур из кейпоинтов
	# эту ересь надо починить 
        if not ballfound:
            for elem in keypoints:
                ballfound = True
                xprev=elem.pt[0]
                yprev=elem.pt[1]
                cv2.circle(frame, (int(elem.pt[0]), int(elem.pt[1])), 5, (33, 255, 237), 2) 
                cv2.putText(frame, 'BALL', (int(elem.pt[0])+10, int(elem.pt[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (33, 255, 237), 1) 
        
        
        
        
#        # shit ends
        
        
        
        # это не нужно, это вывод мяча без всей той телеги выше.
#        for elem in keypoints:
#            x=elem.pt[0]
#            y=elem.pt[1]
#            cv2.circle(frame, (int(x), int(y)), 5, (33, 255, 237), 2) 
#            cv2.putText(frame, 'BALL', (int(x)+10, int(y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (33, 255, 237), 1) 
        
        # почти то же самое, что и пять строк выше
        #frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
         
        # Show keypoints
        cv2.imshow("Keypoints", frame)
        
        
        if cv2.waitKey(25) & 0xFF == 27:
            break
        
    else:
        print('ACHTUNG!')
        break

cap.release()
cv2.destroyAllWindows()