import cv2
import numpy as np
import tensorflow as tf
import imutils
import numpy
import numpy as np
IMAGE_PIXELS=IMAGE_PIXELS = 50 * 50 * 3

def apply_roi(img):
    mask1 = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(0, 680), (110, 448), (505, 215), (784, 210), (1230, 490), (1240, 680)]], dtype=np.int32)
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your img
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask1, roi_corners, ignore_mask_color)
    # apply the mask
    frame = cv2.bitwise_and(img, mask1)
    return frame

def check_with_neural_network(x,y,r,image,sess,logits):
	slice=image[y-r:y+r, x-r:x+r]
	slice = imutils.resize(slice, width=50, height=50)
	imgs=[]
	imgs.append(np.array(slice))
	imgs = np.array(imgs)
	imgs = imgs.reshape(1,IMAGE_PIXELS)
	
	feed_dict={"tf_example:0": imgs}
	predictions = sess.run([tf.nn.softmax(logits)], feed_dict=feed_dict)
	if predictions[0][0][1]>0.8:
		return True
	else:
		return False


def find_blob(img,im,sess,logits):
    params = cv2.SimpleBlobDetector_Params()

    params.filterByConvexity = True
    params.minConvexity = 0.8
        
    params.filterByArea = True
    params.minArea = 5
    #params.maxArea = 300

    params.filterByColor = True
    params.blobColor = 255
 
    #params.filterByCircularity = True
    #params.minCircularity = 0.3

    detector = cv2.SimpleBlobDetector_create(params)
    blobs = detector.detect(img)

    #print("Keypoints # " + str(len(blobs)))
    res=[]
    count=0
    for i in range(len(blobs)):
    	x_ceter=int(blobs[i].pt[0])
    	y_center=int(blobs[i].pt[1])
    	radius=int(blobs[i].size/2)
    	if(check_with_neural_network(x_ceter,y_center,radius,im,sess,logits)):
    		cv2.rectangle(im, (x_ceter-radius, y_center-radius), (x_ceter+radius, y_center+radius), (0, 0, 255), 2)
    		count=count+1
    		res.append((x_ceter,y_center,radius))
    print("Keypoints # " + str(len(blobs))+" #"+str(count))
        #cv2.circle(im, (x_ceter, y_center), radius, (0, 0, 255), 2)
    return res

def surround_windows(image, ball):
	#ищем в радиусе, равном размеру мяча
	xx=ball[0]
	yy=ball[1]
	size=ball[2]
	if(xx-3*size>0):
		area_x_start=xx-3*size
	else:
		area_x_start=0
	if(yy-3*size>0):
		area_y_start=yy-3*size
	else:
		area_y_start=0
	if(xx+3*size<image.shape[1]):
		area_x_finish=xx+size
	else:
		area_x_finish=image.shape[1]
	if(yy+3*size<image.shape[0]):
		area_y_finish=yy+size
	else:
		area_y_finish=image.shape[0]

	for y in range(area_y_start, area_y_finish+1, 2*size): #с шагом = size(половина мяча)
		for x in range(area_x_start, area_x_finish+1, 2*size):
			if((y+2*size<=yy-size) or (x>=xx+size) or (x+2*size<=xx-size) or (y>=yy+size)):
				yield (x, y, image[y:y + 2*size, x:x + 2*size])

def find_surround_balls(old_balls,image,sess,logits):
	res=[]
	for ball in old_balls:
		for (x,y,window) in surround_windows(image,ball):
			if(check_with_neural_network(x+ball[2],y+ball[2],ball[2],image,sess,logits)):
				res.append((x+ball[2],y+ball[2],ball[2]))
	print("New keypoints # "+ str(len(res)))
	return res


cap = cv2.VideoCapture("v2.mp4")
cap.set(cv2.CAP_PROP_POS_MSEC, 100000)#10000
back2 = cv2.createBackgroundSubtractorMOG2(history=100, detectShadows=0)


#ret, frame1 = cap.read()
#prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph('checkpoints/my-model.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/')) 
	logits = tf.get_collection("logits")[0]
	ret, frame = cap.read()
	frame=apply_roi(frame)
	frame1 = back2.apply(frame)
	ball_blobs=find_blob(frame1,frame,sess,logits)
	while (ball_blobs==[]):
		ret, frame = cap.read()
		frame=apply_roi(frame)
		frame1 = back2.apply(frame)
		ball_blobs=find_blob(frame1,frame,sess,logits)
	old_balls=ball_blobs

	while(1):
	    ret, frame = cap.read()
	    frame=apply_roi(frame)
	    frame1 = back2.apply(frame)
	    #if (old_balls==[]):
	    new_balls=find_blob(frame1,frame,sess,logits)
	    #else:
	    	#new_balls=find_surround_balls(old_balls,frame,sess,logits)
	    cv2.imshow("image",frame)

	    k = cv2.waitKey(30) & 0xff
	    if k == 27:
	        break
	    old_balls=new_balls

	cap.release()
	cv2.destroyAllWindows()
