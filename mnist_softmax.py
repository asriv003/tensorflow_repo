from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#one hot vector is a vector which is 0 in most dimensions and 1 in a single dimension.

#evidence = sum( (W_i,j * x_j) + b_i) for all j
#where W_i is weights and b_i is bias for the class i and j is the index for summing over pixels in our input image x

#convert evidence into predicted probabilities 'y' using softmax function

#y = softmax(evidence)

#softmax is exponentiating your inputs and then normanlizing them

#vectorize the procedure
# y = softmax(Wx + b)

import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 784])

#varibale is modifiable tensor that lives in tensorflow graph
#it can be used and even modified by the computation
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#Both variables are intialized as zeros

#model
y = tf.nn.softmax(tf.matmul(x,W) + b)

#implement Cross-Entropy

y_ = tf.placeholder(tf.float32, [None,10])

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#tf.log computes logarihtm of each element of y, then we multiply each element of y_ with corresponding element of tf.log()
#Then tf.reduce_sum add the elements in the second dimension of y, due to reduction_indices=[1] parameter
#Finally tf.reduce_mean computes mean over all examples in the batch
#can be numerically unstable


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#minimize cross_entropy using gradient descent algorithm with learning rate of 0.5

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#Train
for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#Test
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
