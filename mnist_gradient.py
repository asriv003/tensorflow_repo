import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#one hot vector is a vector which is 0 in most dimensions and 1 in a single dimension.

import tensorflow as tf

learning_rate = 0.01
training_epochs = 10
batch_size = 100
display_step = 1

#tf graph input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None,10])


#varibale is modifiable tensor that lives in tensorflow graph
#it can be used and even modified by the computation
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Both variables are intialized as zeros

#model
pred = tf.nn.softmax(tf.matmul(x, W) + b)


#Minimize error using cross entropy
#tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1])) but can give some unstable errors
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

#Gradients calculation
grad_W, grad_b = tf.gradients(xs=[W, b], ys=loss)

new_W = W.assign(W - learning_rate * grad_W)
new_b = b.assign(b - learning_rate * grad_b)


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#minimize cross_entropy using gradient descent algorithm with learning rate of 0.5

init = tf.global_variables_initializer()

with tf.Session() as sess:

	sess.run(init)

	#Train Model
	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(mnist.train.num_examples/batch_size)
		#Loop over all batch
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			_, _, c = sess.run([new_W, new_b, loss], feed_dict={x: batch_xs, y: batch_ys})
			#Average cost
			avg_cost += c/total_batch
		#Display logs per epoch
		if(epoch+1) % display_step == 0:
			print("Epoch: {0} cost: {1}".format((epoch+1), avg_cost))

	print("Optimization Finished")
	
	#Test Model
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	print("Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
