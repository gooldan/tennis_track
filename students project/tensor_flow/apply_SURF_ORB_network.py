import tensorflow as tf
import sys
from PIL import Image
import numpy as np
import PIL
import os
from tensorflow.contrib.session_bundle import exporter
import cv2

def getPoints(image,METHOD):
  image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  if METHOD=="SURF":
    k=10
    surf = cv2.xfeatures2d.SURF_create(k,extended=True,upright = True)
    keypoints, des = surf.detectAndCompute(image,None)
    if(len(keypoints)==0):
      return None
    while (len(keypoints)>10):
        k+=10
        surf = cv2.xfeatures2d.SURF_create(k, extended=True, upright=True)
        keypoints, des = surf.detectAndCompute(image, None)
        if (k>3000):
          return None
    count=des.shape[0]
    array=np.zeros((10, 128))
    for i in range(count):
           for j in range(128):
               array[i][j]=des[i][j]
    return array #10x128

  if METHOD=="ORB":
    orb = cv2.ORB_create()
    kp = orb.detect(image,None)
    if(len(kp)==0):
      return None
    kp, des = orb.compute(image, kp)
    array=np.zeros((500, 32))
    count=des.shape[0]
    for i in range(count):
      for j in range(32):
        array[i][j]=des[i][j]
    return array #500x32


#IMAGE_PIXELS = 1280
IMAGE_PIXELS=16000
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
METHOD="ORB"
#METHOD="SURF"

#print(WORKING_DIR)

image_path = "/home/irina/Desktop/tensor/test.png"
img1 = cv2.imread(image_path)
#image=Image.open(image_path)
#image=image.resize((50, 50), PIL.Image.ANTIALIAS)
SURF_points=getPoints(img1)
if(not(SURF_points is None)):
    imgs=[]
    imgs.append(SURF_points)
    imgs = np.array(imgs)
    imgs = imgs.reshape(1,IMAGE_PIXELS)
    #x = tf.placeholder(tf.float32, shape=(1,IMAGE_PIXELS))

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line 
                       in tf.gfile.GFile("output_labels.txt")]


    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('checkpoints/my-model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./checkpoints/'))
        all_vars = tf.trainable_variables()
        #for v in all_vars:
            #print(v.name)
            #print (sess.run(v.name))
        
        logits = tf.get_collection("logits")[0]
        feed_dict={"tf_example:0": imgs}
        predictions = sess.run([tf.nn.softmax(logits)], feed_dict=feed_dict)
        print(predictions[0][0])
        print(label_lines[1]) if predictions[0][0][0] > predictions[0][0][1] else print(label_lines[0])
else:
    print("Keypoints not found.")





