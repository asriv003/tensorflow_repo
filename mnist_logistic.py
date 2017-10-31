import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#Read Data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Setting parameter
learning_rate = 0.01
batch_size = 128
n_epochs = 25

#Create placeholder
X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])

#Create weight and bias
W = tf.Variable(tf.random_normal(shape=[784,10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros(shape=[1,10]), name="bias")

#Predict Y from X and W,b
logits = tf.matmul(X, W) + b

#Define loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits, Y)
loss = tf.reduce_mean(entropy)

#Define training step
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	
	#Training
	n_batches = int(mnist.train.num_examples/batch_size)
	for i in range(n_epochs):
		for _ in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			sess.run([opt, loss], feed_dict={X: X_batch, Y: Y_batch})

	#Testing