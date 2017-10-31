"""
Siamese implementaition using tensorflow with MNIST example

By:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import os

#local library
import siamese_model
#import visualize

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.01
training_epochs = 10
batch_size = 128

sess = tf.InteractiveSession()

#setup siamese network
siamese = siamese_model.siamese()
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(siamese.loss)
#initialize varaibles
tf.global_variables_initializer()

for step in range(10000):
	batch_x1, batch_y1 = mnist.train.next_batch(batch_size)
	batch_x2, batch_y2 = mnist.train.next_batch(batch_size)
	batch_y = (batch_y1 == batch_y2).astype('float')

	_, loss_v = sess.run([train_step, siamese.loss], feed_dict={
													siamese.x1: batch_x1,
													siamese.x2: batch_x2,
													siamese.y_: batch_y})

	if np.isnan(loss_v):
		print("Model diverged with loss = NaN")
		quit()

	if step % 10 == 0:
		print('step %d: loss %.3f' % (step, loss_v))

	if step % 1000 == 0 and step > 0:
		embed = siamese.o1.eval({siamese.x1: mnist.test.images})