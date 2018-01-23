from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

import time
import math
import numpy
import numpy as np
import random
from PIL import Image
import PIL
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import cv2

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
#flags.DEFINE_integer('hidden1', 162, 'Number of units in hidden layer 1.')
#flags.DEFINE_integer('hidden2', 18, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('hidden1', 800, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 40, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')
NUM_CLASSES = 2
#IMAGE_PIXELS=1280 #Input
IMAGE_PIXELS=16000
METHOD="ORB"
#METHOD="SURF"

class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      
      #images = images.reshape(images.shape[0],images.shape[1],1)
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

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
      #408 None element

    	#image2 = cv2.drawKeypoints(image,keypoints,None,(255,0,0),2)
    	#print("Количество keypoints:")
    	#print(len(keypoints))
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

def getMaxLen(image_list):
  maxLen=0
  for i in range(len(image_list)):
      #print(image_list[i])
      k=10
      image=cv2.imread(image_list[i])
      surf = cv2.xfeatures2d.SURF_create(k,extended=True,upright = True)
      keypoints, des = surf.detectAndCompute(image,None)
      while (len(keypoints)>10):
        k+=20
        surf = cv2.xfeatures2d.SURF_create(k,extended=True,upright = True)
        keypoints, des = surf.detectAndCompute(image,None)
        if (k>3000):
          break
      if(len(keypoints)>maxLen):
          maxLen=len(keypoints)
  return maxLen

def read_data_sets():
    class DataSets(object):
        pass
    data_sets = DataSets()
    dtype=tf.float32
    print("Preparing images..")
    # Get the sets of images and labels for training, validation, and
    with open('inf.info') as f:
        image_list=[]
        label_list=[]
        i=0
        while(True):
            i=i+1
            line=f.readline()
            if line=='':
                break;
            x=int(line.split()[2])
            y=int(line.split()[3])
            xx=int(line.split()[4])
            yy=int(line.split()[5])
            im = Image.open(line.split()[0])
            label=int(line.split()[1])
           	#img=im.crop((x,y,xx,yy))
            #img = img.resize((50, 50), PIL.Image.ANTIALIAS)
            can=False
            if label==1:
              #img.save('cropped_n_resized/pos/_'+line.split()[0].split('/')[1])
              image=cv2.imread('cropped_n_resized/pos/_'+line.split()[0].split('/')[1])
              if(not(getPoints(image,METHOD)is None)):
                image_list.append('cropped_n_resized/pos/_'+line.split()[0].split('/')[1])
                can=True
              # else:
              #   im=np.array(image)
              #   print(im.shape)
            if label==0:
              #img.save('cropped_n_resized/neg/_'+str(i)+'_'+line.split()[0].split('/')[1])
              image=cv2.imread('cropped_n_resized/neg/_'+str(i)+'_'+line.split()[0].split('/')[1])
              if(not(getPoints(image,METHOD) is None)):
                image_list.append('cropped_n_resized/neg/_'+str(i)+'_'+line.split()[0].split('/')[1])
                can=True
              # else:
              #   im=np.array(image)
              #   print(im.shape)
            if(can):
              label_list.append(label)
        #shuffle lists together
        combined = list(zip(image_list, label_list))
        random.shuffle(combined)
        image_list[:], label_list[:] = zip(*combined)
   
    print(len(image_list))
    total_batch_size=len(image_list)
    train_images = []
    test_images = []
    train_labels=[]
    test_labels=[]
    n=total_batch_size-int(0.2*total_batch_size) #20% for test images
    #maxLenKp=getMaxLen(image_list)
    #print(maxLenKp) #=78

    for i in range(len(image_list)):
      image=cv2.imread(image_list[i])
      SURF_points=getPoints(image,METHOD)
      if(i<n):    
        #image = image.resize((150,200))
        train_images.append(SURF_points)
        train_labels.append(label_list[i])
      else:
        test_images.append(SURF_points)
        test_labels.append(label_list[i])
    
    train_images2 = np.array(train_images,dtype=np.uint8)
    test_images2 = np.array(test_images,dtype=np.uint8)
    train_images2 = train_images2.reshape(len(train_images),IMAGE_PIXELS)
    test_images2 = test_images2.reshape(len(test_images),IMAGE_PIXELS)

    train_labels = np.array(train_labels,dtype=np.uint8)
    test_labels = np.array(test_labels,dtype=np.uint8)

    data_sets.train = DataSet(train_images2, train_labels, dtype=dtype)
    data_sets.test = DataSet(test_images2, test_labels, dtype=dtype)

    return data_sets

def inference(images, hidden1_units, hidden2_units):
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
    tf.add_to_collection("logits", logits)
  return logits


def cal_loss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def training(loss, learning_rate):
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(batch_size):

  images_placeholder = tf.placeholder(tf.float32, shape=(None,IMAGE_PIXELS),name='tf_example')
  labels_placeholder = tf.placeholder(tf.int32)
  return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl):
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)

  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  #Runs one evaluation against the full epoch of data.
  #And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))



def run_training():
  data_sets = read_data_sets()
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    images_placeholder, labels_placeholder = placeholder_inputs(FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = inference(images_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = cal_loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    #summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.
    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,
    #                                        graph_def=sess.graph_def)

    # And then after everything is built, start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      #feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)
      epoch_loss = 0
      for _ in range(int(data_sets.train.num_examples/FLAGS.batch_size)):
                epoch_x, epoch_y = data_sets.train.next_batch(FLAGS.batch_size)
                _, loss_value = sess.run([train_op, loss], feed_dict={images_placeholder: epoch_x, labels_placeholder: epoch_y})
                epoch_loss += loss_value
      #_, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)
      duration = time.time() - start_time
      print('Step: ', step, 'completed out of ',FLAGS.max_steps,'loss:',epoch_loss, '(',duration,')')
      #if step % 50 == 0:
        # Print status to stdout.
        #print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        #summary_str = sess.run(summary_op, feed_dict=feed_dict)
        #summary_writer.add_summary(summary_str, step)

      # Save a checkpoint and evaluate the model periodically.  
      if (step + 1) % 500 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, FLAGS.train_dir, global_step=step)
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.train)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                data_sets.test)   
        
    saver.save(sess, "my-model")

    with gfile.FastGFile('/home/irina/Desktop/tensor/output_graph.pb', 'wb') as f:
      f.write(sess.graph_def.SerializeToString())
    with gfile.FastGFile('/home/irina/Desktop/tensor/output_graph.pb', 'w') as f:
      f.write(str(sess.graph_def))
    with gfile.FastGFile('/home/irina/Desktop/tensor/output_labels.txt', 'w') as f:
      f.write('ball' + '\n')
      f.write('not ball')


def main(_):
  run_training()
if __name__ == '__main__':
  tf.app.run()