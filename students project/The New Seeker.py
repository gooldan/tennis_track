import cv2
import numpy as np
from matplotlib import pyplot as plt

def dist(x,y,X,Y):
    return np.sqrt((x-X)**2 + (y-Y)**2)

cap = cv2.VideoCapture('C:\\Users\\ф\\Documents\\test\\Tennis1.mp4')
back = cv2.imread('Background.png', 0)
back2 = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=0)
start=470000
counter= start
cap.set(cv2.CAP_PROP_POS_MSEC, start)
ret, frame1 = cap.read()
perFrame=False
#baseFrame=np.zeros_like(frame)
frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
while True:
    ret, frame2 = cap.read()
    
    # applying mask
    mask = np.zeros(frame2.shape, dtype=np.uint8)
    roi_corners = np.array([[(0,720),(0,460),(480,230),(800,230),(1280,460),(1280,720)]], dtype=np.int32)
    channel_count = frame2.shape[2]  # i.e. 3 or 4 depending on your img
    ignore_mask_color = (255,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    #frame1 = cv2.bitwise_and(frame1, mask)
    frame2 = cv2.bitwise_and(frame2, mask)

    
    #frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    
    
    
    
    
    diff = cv2.subtract(frame2, back)
    
    
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), dtype = np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=4)
    thresh = cv2.dilate(thresh, kernel, iterations=4)
    _, cont, hier = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)    
    
    
    # ищем контур с максимальной площадью
    # смею предположить, что это туловище игрока
    maxArea = 0
    for elem in cont:
        area = cv2.contourArea(elem)
        if area > maxArea:
            maxArea = area
            maxCnt = elem
            
    # ищем центр масс тела
    M = cv2.moments(maxCnt)
    Cx = int(M['m10']/M['m00'])
    Cy = int(M['m01']/M['m00'])
    # предполагаемый контур игрока
    manCnt = []
    
    # выбираем контуры, находящиеся неподалёку от контура туловища
    for elem in cont:
        Mm = cv2.moments(elem)
        if Mm['m00'] == 0:
            continue
        ccx = int(Mm['m10']/Mm['m00'])
        ccy = int(Mm['m01']/Mm['m00'])
        if dist(Cx, Cy, ccx, ccy) <= 200:
            manCnt.append(elem)
    
    # находим два самых нижних контура среди различных контуров игрока
    minx1, miny1, minx2, miny2 = 0, 0, 0, 0
    for cont in manCnt:
        M = cv2.moments(cont)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if cy > miny1:
            minx2, miny2 = minx1, minx2
            minx1, miny1 = cx, cy
        elif cy > miny2:
            minx2, miny2 = cx, cy
    
    
    
    
    
    
    
    diff1 = back2.apply(frame1)
    _, diff1 = cv2.threshold(diff1,10,255, cv2.THRESH_BINARY_INV)
    diff2 = back2.apply(frame2)
    _, diff2 = cv2.threshold(diff2,10,255, cv2.THRESH_BINARY_INV)
    
        
    params = cv2.SimpleBlobDetector_Params()

    params.filterByConvexity = True
    params.minConvexity = 0.8
    
    #params.filterByInertia = True
    #params.minInertiaRatio = 0.3
        
    params.filterByArea = True
    params.minArea = 15
    #params.maxArea = 300
 
    #params.filterByCircularity = True
    #params.minCircularity = 0.3


    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints1 = detector.detect(diff1)
    keypoints2 = detector.detect(diff2)
    
    res1 = []
    res2 = []
    for elem in keypoints1:
            x=elem.pt[0]
            y=elem.pt[1]
            for thing in keypoints1:
                if dist(thing.pt[0], thing.pt[1], x, y) > 150:
                    res1.append(thing)
    for elem in keypoints2:
            x=elem.pt[0]
            y=elem.pt[1]
            for thing in keypoints2:
                if dist(thing.pt[0], thing.pt[1], x, y) > 150:
                    res2.append(thing)
    keypoints1.clear()
    keypoints2.clear()
    for elem in res1:
        move=True
        for thing in res2:
            if np.sqrt((elem.pt[0]-thing.pt[0])*(elem.pt[0]-thing.pt[0])+(elem.pt[1]-thing.pt[1])*(elem.pt[1]-thing.pt[1])) < 1:
                move=False
                break
        if move==True:
            keypoints2.append(elem)
            
            
            
           
    xnoga = (minx1+minx2)/2
    ynoga = (miny1+miny2)/2
    
    if len(keypoints2)!=0:
    
        ball=keypoints2.copy()
        distanse=0
        for elem in keypoints2:
            if (dist(elem.pt[0], elem.pt[1], xnoga, ynoga) > distanse) and (np.sqrt((elem.pt[1] - ynoga)*(elem.pt[1] - ynoga)) > 150):
                distanse =dist(elem.pt[0], elem.pt[1], xnoga, ynoga)
                ball.clear()
                ball.append(elem)
    
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)
        ball1=ball.pop()
        x=ball1.pt[0]
        y=ball1.pt[1]
        cv2.circle(frame1, (int(x), int(y)), 5, (33, 255, 237), 2) 
        cv2.putText(frame1, 'BALL', (int(x)+10, int(y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (33, 255, 237), 1)
        #for elem in keypoints1:
        #        x=elem.pt[0]
        #        y=elem.pt[1]
        #        cv2.circle(frame1, (int(x), int(y)), 5, (33, 255, 237), 2) 
        #        cv2.putText(frame1, 'BALL', (int(x)+10, int(y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (33, 255, 237), 1)
        #        
        diff1 = cv2.cvtColor(diff1, cv2.COLOR_GRAY2BGR)
        x=elem.pt[0]
        y=elem.pt[1]
        cv2.circle(diff1, (int(x), int(y)), 5, (0, 150, 0), 2) 
        cv2.putText(diff1, 'BALL', (int(x)+10, int(y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1)
        #for elem in keypoints1:
        #        x=elem.pt[0]
        #        y=elem.pt[1]
        #        cv2.circle(diff1, (int(x), int(y)), 5, (0, 150, 0), 2) 
        #        cv2.putText(diff1, 'BALL', (int(x)+10, int(y)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 0), 1)
        
    cv2.circle(frame1, (minx1, miny1), 5, (0, 0, 255), -1)
    cv2.circle(frame1, (minx2, miny2), 5, (0, 0, 255), -1)
    cv2.circle(frame1, (Cx, Cy), 5, (0, 150, 0), -1)
    cv2.line(frame1, (Cx, Cy), (minx1, miny1), (255, 0, 0), 2)
    cv2.line(frame1, (Cx, Cy), (minx2, miny2), (255, 0, 0), 2)
    cv2.putText(frame1, 'frame #'+str(start), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
    
    cv2.imshow("asdasd2",diff1)
    
    cv2.imshow("asdasd1",frame1)
    
    frame1=frame2     
    
    
    k = cv2.waitKey(5)
    if(k!=-1):
        print (k)
    if k == 27:
        break
    if k == 13:
        cv2.imwrite('pic.png', frame)
        
    if k == 32 or perFrame==True:
        k = cv2.waitKey(0)
        if(k==32):
            perFrame=False
            continue
        if(k == 2555904):
            perFrame=True
            continue
cap.release()
cv2.destroyAllWindows()