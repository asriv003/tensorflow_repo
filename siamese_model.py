import tensorflow as tf

class siamese:

	#create model
	def __init__(self):
		#input vector
		self.x1 = tf.placeholder(tf.float32, [None, 784])
		self.x2 = tf.placeholder(tf.float32, [None, 784])

		with tf.variable_scope("siamese") as scope:
			#creating NN using inputs
			self.o1 = self.network(self.x1)
			scope.reuse_variables()
			self.o2 = self.network(self.x2)

		#create loss
		self.y_ = tf.placeholder(tf.float32, [None,10])
		self.loss = self.loss_with_spring()


	#Neural Network Architecture
	def network(self, x):
		weights = []

		fc1 = self.fc_layer(x, 1024, "fc1")
		ac1 = tf.nn.relu(fc1)
		fc2 = self.fc_layer(ac1, 1024, "fc2")
		ac2 = tf.nn.relu(fc2)
		fc3 = self.fc_layer(ac2, 2, "fc3")
		return fc3

	#Fully connected layer
	def fc_layer(self, bottom, n_weight, name):
		assert len(bottom.get_shape()) == 2
		n_prev_weight = bottom.get_shape()[1]
		w_initer = tf.truncated_normal_initializer(stddev=0.01)
		b_initer = tf.constant(0.01, shape=[n_weight], dtype=tf.float32)
		W = tf.get_variable(name+'W', dtype=tf.float32, shape=[n_prev_weight, n_weight], initializer=w_initer)
		b = tf.get_variable(name+'b', dtype=tf.float32, initializer=b_initer)
		fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
		return fc

	#loss function
	def loss_with_spring(self):
		margin = 5.0
		#true labels
		labels_t = self.y_
		#fail labels
		labels_f = tf.subtract(1.0, self.y_, name="1-y_i")
		#calculating eucledian distance
		eucd2 = tf.pow(tf.subtract(self.o1, self.o2),2)
		eucd2 = tf.reduce_sum(eucd2, 1)
		eucd = tf.sqrt(eucd2+1e-6, name="eucd")
		C = tf.constant(margin, name="C")
		# ( y_i * ||CNN(p1_i) - CNN(p2_i)||^2 ) + ( 1-y_i * (max(0, C - ||CNN(p1_i) - CNN(p2_i)||))^2 )
		pos = tf.multiply(labels_t, eucd2, name="y_i_x_eucd2")
		neg = tf.multiply(labels_f, tf.pow(tf.maximum(0.0, tf.subtract(C, eucd)), 2), name="Ny_i_x_C-eucd_xx_2")
		cumm_losses = tf.add(pos, neg, name="cumm_losses")
		loss = tf.reduce_mean(cumm_losses, name="loss")
		return loss

	#loss funtion with step
	def loss_with_step(self):
		margin = 5.0
		#true labels
		labels_t = self.y_
		#fail labels
		labels_f = tf.subtract(1.0, self.y_, name="1-y_i")
		#calculating eucledian distance
		eucd2 = tf.pow(tf.subtract(self.o1, self.o2),2)
		eucd2 = tf.reduce_sum(eucd2, 1)
		eucd = tf.sqrt(eucd2+1e-6, name="eucd")
		C = tf.constant(margin, name="C")
		pos = tf.multiply(labels_t, eucd, name="y_x_eucd")
		neg = tf.multiply(labels_f, tf.maximum(0, tf.subtract(C, eucd)), name="Ny_x_C-eucd")
		cumm_losses = tf.add(pos, neg, name="cumm_losses")
		loss = tf.reduce_mean(cumm_losses, name="loss")
		return loss
