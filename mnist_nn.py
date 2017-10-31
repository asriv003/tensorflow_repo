
#Multilayer Convolutional Neural Network
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

#To initialize weights and biases
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#convolution operation
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#pooling operation
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


#First Convolution Layer
x = tf.placeholder(tf.float32, [None, 784])
#Convolution will compute 32 features for each 5x5 patch
#[5, 5, 1, 32] -> first two are patch size, next is number of input channels, last is number of output channels
W_conv1 = weight_variable([5, 5, 1, 32])
#bias vector
b_conv1 = bias_variable([32])

#To apply layer, reshape x to 4d tensor, with 2nd and 3rd dimension equal to width and height of image and final dimension to number of color channels

x_image = tf.reshape(x, [-1, 28, 28, 1])

#Convolve x_image with weight tensor, add the bias, apply the ReLu function
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

#Apply max pool
h_pool1 = max_pool_2x2(h_conv1)
#Image size has been reduced to 14x14

#Second Convolution Layer

W_conv2 = weight_variable([5, 5, 32, 64])
#bias vector
b_conv2 = bias_variable([64])

#Convolve x_image with weight tensor, add the bias, apply the ReLu function
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

#Apply max pool
h_pool2 = max_pool_2x2(h_conv2)

#Image size has been reduced to 7x7

#Densily Connected layer
#Add fully connected layer to allow procession on entire image
#Reshape tensor from polling layer into batch of vectors
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

#Multiply by weigth matrix and add bias and apply ReLu
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout
#To reduce overfitting apply dropout before readout layer
#turn dropout on during training and turn it off during testing

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

y_ = tf.placeholder(tf.float32, [None,10])

#Evaluate Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#Use ADAM optimizer instrad of Gradient Descent optimizer
#pass additonal 'keep_prob' parameter in feed_dict for dropout rate
#Add log to every 100th iteration in trainng process

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	for i in range(20000):
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
			print('step: %d, training accuracy: %f' % (i, train_accuracy))
		train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
		
	print('test accuracy: %f' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
