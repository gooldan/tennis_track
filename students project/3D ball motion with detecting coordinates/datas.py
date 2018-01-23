import cv2
import numpy as np
import imutils

img1 = cv2.imread('tabletennis01660/0pic001662.jpg', cv2.IMREAD_COLOR)
im1 = cv2.imread('tabletennis01660/1pic001662.jpg')

# img2 = cv2.imread('tabletennis01660/0pic001663.jpg')
# im2 = cv2.imread('tabletennis01660/1pic001663.jpg')

img3 = cv2.imread('tabletennis01660/0pic001664.jpg')
im3 = cv2.imread('tabletennis01660/1pic001664.jpg')

img6 = cv2.imread('tabletennis01660/0pic001667.jpg')
im6 = cv2.imread('tabletennis01660/1pic001667.jpg')

img1 = cv2.imread('tabletennis01660/0pic001668.jpg')
im7 = cv2.imread('tabletennis01660/1pic001668.jpg')

img8 = cv2.imread('tabletennis01660/0pic001669.jpg')
im8 = cv2.imread('tabletennis01660/1pic001669.jpg')

img10 = cv2.imread('tabletennis01660/0pic001671.jpg')
im10 = cv2.imread('tabletennis01660/1pic001671.jpg')

img11 = cv2.imread('tabletennis01660/0pic001672.jpg')
im11 = cv2.imread('tabletennis01660/1pic001672.jpg')

cv2.imshow("im", img1)

mask = np.zeros(img1.shape, dtype=np.uint8)
roi_corners = np.array([[(120,0),(120,30),(180,30),(180,0)]], dtype=np.int32)
channel_count = img1.shape[2]
ignore_mask_color = (255,)*channel_count
cv2.fillPoly(mask, roi_corners, ignore_mask_color)


img1=img1[100:img1.shape[0],200:img1.shape[1]]

resized = imutils.resize(img1, width=int(img1.shape[1] * 5))
resized2 = imutils.resize(img1, width=int(img1.shape[1] * 5))


refPt = []
drawing = False
drawing2 = False
center=[]

def click_circle(event, x, y, flags, param):
    global drawing, rx,ry, ex,ey,center,pictured

    if event == cv2.EVENT_LBUTTONDOWN:
        rx, ry=x,y
        drawing = True

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            #clone = resized.copy()
            ex, ey = x, y
            cv2.circle(resized, (int(abs((ex-rx)/2))+rx, int(abs((ey-ry)/2))+ry), int(abs((ex-rx)/2)), (0, 255, 0), 1)
            cv2.imshow("image", resized)

    elif event == cv2.EVENT_LBUTTONUP:
        ex, ey = x, y
        drawing = False
        print(int((abs(refPt[1][0]-refPt[0][0])/2+rx)/5),int((abs(refPt[1][1]-refPt[0][1])/2+ry)/5))
        cv2.circle(resized, (int(abs((ex-rx)/2))+rx, int(abs((ey-ry)/2))+ry), int(abs((ex-rx)/2)), (0, 255, 0), 1)
        center.append(int(abs((ex-rx)/2))+rx)
        center.append(int(abs((ey-ry)/2))+ry)
        cv2.imshow("image", resized)

def click_rectangle(event, x, y, flags, param):
    global refPt, drawing2, rx2, ry2, ex2, ey2

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        rx2, ry2 = x, y
        drawing2 = True

    # if event == cv2.EVENT_MOUSEMOVE:
    #     if drawing2 == True:
    #         ex2, ey2 = x, y
    #         cv2.rectangle(resized2, (rx2,ry2), (ex2,ey2), (255, 0, 0), 1)
    #         cv2.imshow("image2", resized2)

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        #ex2, ey2 = x, y
        drawing2 = False
        print(int((abs(refPt[1][0] - refPt[0][0]) / 2 + rx2)/5)+200, int((abs(refPt[1][1] - refPt[0][1]) / 2 + ry2)/5)+100)
        cv2.rectangle(resized2, refPt[0], refPt[1], (255, 0, 0), 1)
        cv2.imshow("image2", resized2)



#draw circle
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", click_circle)

#draw rectangle
cv2.namedWindow("image2")
cv2.setMouseCallback("image2", click_rectangle)
while(1):
    #cv2.imshow('image',resized)
    cv2.imshow('image2',resized2)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
