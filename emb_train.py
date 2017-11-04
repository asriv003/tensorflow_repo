"""
Siamese graph embedding implementaition using tensorflow

By:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#to use tfdbg
#wrap session object with debugger wrapper
from tensorflow.python import debug as tf_debug

import tensorflow as tf
import numpy as np
import os

#local library
import siamese_emb

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


#row size
row_size = 5
#feature vector size(m)
vector_size = 8
#dimension of desired embedding(p)
emb_size = 10

learning_rate = 0.01

#To initialize weights and biases
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

x_1 = np.random.rand(5,8)
x_2 = np.random.rand(5,8)


def train_siamese():
	with tf.Graph().as_default():
		#init class
		siamese = siamese_emb.siamese(row_size=row_size, vector_size=vector_size, emb_size=emb_size)

		global_step =  tf.Variable(0, trainable=False)

		
		#can use other optimizers
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		#train_op = optimizer.minimize(siamese.loss, global_step=global_step)

		init_op = tf.global_variables_initializer()
		init_local = tf.local_variables_initializer()

		sess = tf.Session()


		with sess.as_default() as sess:
			writer = tf.summary.FileWriter('./graphs', sess.graph)
			sess.run(init_op)
			sess.run(init_local)
			#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
			#show which variables are going to be train
			variable_names = [v.name for v in tf.trainable_variables()]
			values = sess.run(variable_names)
			print("Trainable Varaible List:")
			for k, v in zip(variable_names, values):
				print("Variable: ", k, ", Shape: ", v.shape)

			#sess.run([train_op, siamese.loss], feed_dict={siamese.x1= x_1, siamese.x2 = x_2})
			r1 = sess.run([siamese.loss], feed_dict={siamese.x1: x_1, siamese.x2: x_2, siamese.y: [1] })
			print(r1)
			#print(r2)
			writer.close()
			#print(r3)
'''
with tf.Session() as sess:
	#setup siamese network
	siamese = Siamese(row_size=row_size, vector_size=vector_size, emb_size=emb_size) 
	sess.run(tf.global_variables_initializer())
	#print(x_1)
	#print(sess.run(x_1))
	#print(x_2)
	# graph_emb1, graph_emb2 = sess.run([siamese.e1, siamese.e2], feed_dict={
	# 												siamese.x1: x_1, siamese.x2: x_2})

	graph_emb1,W_1,W_2 = sess.run(siamese.e1, feed_dict={siamese.x1: x_1 })
	print(graph_emb1)
	print(W_1)
	print(W_2)
	#print(graph_emb2)

# #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(siamese.loss)
# #initialize varaibles
# tf.global_variables_initializer()



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
'''

if __name__ == "__main__":
	train_siamese()