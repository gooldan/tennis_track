import cv2
import numpy as np
import math

def dist(x,y,X,Y):
    return math.sqrt((x-X)**2 + (y-Y)**2)

cap = cv2.VideoCapture('Main2.mp4')
cap.set(cv2.CAP_PROP_POS_MSEC,450000)
back = cv2.imread('back.jpg', 0)
back2 = cv2.imread('frame21.jpg', 0)

paused = False
x, y = -1, -1

while True:
    
    if not paused:
        
        ret, frame = cap.read()
        
        if ret == True:
            
            if not paused:
                
                # setting a ROI
                mask = np.zeros(frame.shape, dtype=np.uint8)
                roi_corners = np.array([[(0,720),(0,460),(440,280),(880,280),(1280,460),(1280,720)]], dtype=np.int32)
                channel_count = frame.shape[2]  # i.e. 3 or 4 depending on your img
                ignore_mask_color = (255,)*channel_count
                cv2.fillPoly(mask, roi_corners, ignore_mask_color)
                Frame = cv2.bitwise_and(frame, mask)
                
                Frame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
                
                # background subtraction
                diff = cv2.subtract(Frame, back2)
                
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
                    
                    # убираем элементы с непропорционально большими/малыми высотами, т.е. линии
                    _,_,w,h = cv2.boundingRect(elem)
                    if 0.5 < w/h < 3:
                        if dist(Cx, Cy, ccx, ccy) <= 200:
                            manCnt.append(elem)
                Cont = cv2.drawContours(thresh, manCnt, -1, 60, -1)
                
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
                        
                # обработка случая, когда одна нога находится за другой
                if abs(minx1-minx2) < 10:
                    minx2 = minx1
                    miny2 = miny1
                    cv2.putText(frame, '!!!', (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                
                # выводим результаты в окно с контурами
                Thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                cv2.circle(Thresh, (Cx, Cy), 5, (0, 150, 0), -1)
                cv2.circle(Thresh, (minx1, miny1), 5, (0, 0, 255), -1)
                cv2.circle(Thresh, (minx2, miny2), 5, (0, 0, 255), -1)
                cv2.line(Thresh, (Cx, Cy), (minx1, miny1), (255, 0, 0), 2)
                cv2.line(Thresh, (Cx, Cy), (minx2, miny2), (255, 0, 0), 2)
                
                # выводим результаты в окно с исходной картинкой
                cv2.circle(frame, (minx1, miny1), 5, (0, 0, 255), -1)
                cv2.circle(frame, (minx2, miny2), 5, (0, 0, 255), -1)
                cv2.circle(frame, (Cx, Cy), 5, (0, 150, 0), -1)
                cv2.line(frame, (Cx, Cy), (minx1, miny1), (255, 0, 0), 2)
                cv2.line(frame, (Cx, Cy), (minx2, miny2), (255, 0, 0), 2)

    cv2.imshow('diff', diff)
    cv2.imshow('Cont', Cont)
    cv2.imshow('thresh', Thresh)
    cv2.imshow('frame', frame)

    k = cv2.waitKey(2)

    if k & 0xFF == ord(' '):
        paused = not paused
    elif k & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()