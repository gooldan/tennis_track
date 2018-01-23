import tensorflow as tf
import sys
from PIL import Image
import numpy as np
import PIL
import os
from tensorflow.contrib.session_bundle import exporter

IMAGE_PIXELS = 50 * 50 * 3
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
#print(WORKING_DIR)

image_path = "/home/irina/Desktop/tensor/test.pngs"

#image_data = tf.gfile.FastGFile(image_path, 'rb').read()
image=Image.open(image_path)
image=image.resize((50, 50), PIL.Image.ANTIALIAS)
imgs=[]
imgs.append(np.array(image))
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






