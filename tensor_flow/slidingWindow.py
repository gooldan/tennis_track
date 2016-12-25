import time
import cv2
import imutils
import numpy
import numpy as np
import tensorflow as tf

path="pos/2.jpg"
IMAGE_PIXELS = 50 * 50 * 3

def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def window_size(image, scale=1.5, minSize=(20, 20), maxSize=(50,50)):
	yield minSize
	while (1):
		#(minSize[0], minSize[1])=(minSize[0]*scale, minSize[1]*scale)
		minSize=(int(minSize[0]*scale), int(minSize[1]*scale))
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0] or minSize[0]>maxSize[0] or minSize[1]>maxSize[1]:
			break
		yield minSize

with tf.Session() as sess:
	new_saver = tf.train.import_meta_graph('my-model.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./')) 
	logits = tf.get_collection("logits")[0]
	image = cv2.imread(path)
	for (winW, winH) in window_size(image):
		print(winW, winH)
		for (x, y, window) in sliding_window(image, stepSize=int(winW/2), windowSize=(winW, winH)):
				# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue
			clone = image.copy()		
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

			img=image[y:y+winH, x:x+winW]
			img = imutils.resize(img, width=50, height=50)
			imgs=[]
			imgs.append(np.array(img))
			imgs = np.array(imgs)
			imgs = imgs.reshape(1,IMAGE_PIXELS)
			feed_dict={"tf_example:0": imgs}
			predictions = sess.run([tf.nn.softmax(logits)], feed_dict=feed_dict)
			    #print(predictions[0][0])
			if predictions[0][0][0]>0.8:
			    cv2.rectangle(image, (x, y), (x + winW, y + winH), (0, 0, 255), 2)

			cv2.imshow("Window", clone)
			cv2.waitKey(1)
			#time.sleep(0.0025)
