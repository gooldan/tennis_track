import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy as sp

def find_if_close(cnt1,cnt2,const):
    M1 = cv2.moments(cnt1)
    M2 = cv2.moments(cnt2)
    
    if(M1["m00"]!=0 and M2["m00"]!=0):
        cX1 = int((M1["m10"] / M1["m00"]))
        cY1 = int((M1["m01"] / M1["m00"]))
        cX2 = int((M2["m10"] / M2["m00"]))
        cY2 = int((M2["m01"] / M2["m00"]))
        
        dist = np.sqrt((cX1-cX2)*(cX1-cX2)+(cY1-cY2)*(cY1-cY2)) 
        if dist < const :
            return True,(cX1,cY1),(cX2,cY2)
    return False,(0,0),(0,0)        

#cap = cv2.VideoCapture('vid.mp4')
#cap = cv2.VideoCapture('20160519_081947.mp4')

cap = cv2.VideoCapture("D:\\vidsnew\\vid_last.mp4")
#cap.set(cv2.CAP_PROP_POS_MSEC,161000)
cap.set(cv2.CAP_PROP_POS_MSEC,870000)
fgbg = cv2.createBackgroundSubtractorMOG2(500,160,False)
perFrame=False
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
iii = 0
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
oldUnified=[]
unified = []
linearr=[]
counter=0
while(1):
    iii+=1
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    black = np.zeros_like(fgmask)
    res=np.zeros_like(frame)
    #res = cv2.bitwise_and(frame,frame,mask = fgmask)
    edges = cv2.Canny(fgmask,200,200)
    #fgmask = cv2.dilate(fgmask, kernel1, iterations=5)

    ret,thresh = cv2.threshold(fgmask,127,255,cv2.THRESH_BINARY)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20,50,50])
    upper_yellow = np.array([50,255,255])
    maskcol = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res1 = cv2.bitwise_and(res,res,mask = maskcol)
    
    #######
    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))

    for i,cnt1 in enumerate(contours):
        x = i    
        if i != LENGTH-1:
            for j,cnt2 in enumerate(contours[i+1:]):
                x = x+1
                dist,bbbb,bbbbbb = find_if_close(cnt1,cnt2,50)
                if dist == True:
                    val = min(status[i],status[x])
                    status[x] = status[i] = val
                else:
                    if status[x]==status[i]:
                        status[x] = i+1
    sizeold=len(linearr)
    if(len(status>0)):
        oldUnified=list(unified)
        unified=[]
        maximum = int(status.max())+1
        for i in range(maximum):
            pos = np.where(status==i)[0]
            if pos.size != 0:
                cont = np.vstack(contours[i] for i in pos)
                hull = cv2.convexHull(cont)
                area=cv2.contourArea(hull)
                if(area<380):
                    unified.append(hull)
        
        if(len(oldUnified)>0):
            for elem1 in unified:
                for elem2 in oldUnified:
                    if elem2 is not None:
                        #print(str(elem1))
                        dist,center1,center2 = find_if_close(elem1,elem2,70)
                        if dist == True:
                            linearr.append((center1,center2));
                            
                            
        cv2.drawContours(black,unified,-1, (255,255,255), -1)
        cv2.drawContours(frame,unified,-1, (0,255,0), 2)
      
        res = cv2.bitwise_and(frame,frame,mask = black)
    if(sizeold==len(linearr)):
        counter+=1
    else:
        counter=0    
    if(counter==10):    
        linearr=[]
        counter=0
    for center1,center2 in linearr:
        cv2.line(frame,center1,center2,(0,0,255),2)
    cv2.imshow('asdasdf',fgmask)
    cv2.imshow('asdasdff',frame)
    
    delay = 1
    k = cv2.waitKey(delay)
    if(k!=-1):
        print (k)
    if k == 27:
        break
    if k == 112:
        cv2.imwrite('pic{:>05}.jpg'.format(iii), res)
    if k == 32 or perFrame==True:
        k = cv2.waitKey(0)
        if(k==32):
            perFrame=False
            continue
        if k == 112:
            cv2.imwrite('pic{:>05}.jpg'.format(iii), res)
            cv2.imwrite('picr{:>05}.jpg'.format(iii), res1)
        if(k == 2555904): 
            perFrame=True
            continue
    
        
    

cap.release()
cv2.destroyAllWindows()