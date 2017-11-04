import tensorflow as tf

class siamese:

	#calculate embedding
	def emb_generation(self, x, row_size, vector_size, emb_size, name):
		#print(x)
		with tf.name_scope(name):

			#embeddings to be calculated
			mu_v_0 = tf.Variable(tf.zeros([row_size, emb_size], dtype=tf.float32), name="mu_v_0", trainable=False)
			mu_v_1 = tf.Variable(tf.zeros([row_size, emb_size], dtype=tf.float32), name="mu_v_1", trainable=False)


			#learnable parameters
			#summation of embedding variable(rxp)
			l_v = tf.Variable(tf.zeros([row_size, emb_size], dtype=tf.float32), name="l_v", trainable=False)
			w_init = tf.truncated_normal_initializer(stddev=0.1)
			#W_1 = tf.Variable()
			W_1 = tf.get_variable('W_1', dtype=tf.float32, shape=[vector_size, emb_size], initializer=w_init)
			W_2 = tf.get_variable('W_2', dtype=tf.float32, shape=[emb_size, emb_size], initializer=w_init)
			P = tf.get_variable('P_relu', dtype=tf.float32, shape=[emb_size, emb_size], initializer=w_init)
			#pr = tf.Print(W_1, [W_1])
			#pr.eval()

			#Running T times
			for t in range(20):
				#calculating summation of neighbour vertexes
				#print(t)
				l_temp = tf.reduce_sum(mu_v_0, axis=0)
				l_tile = tf.tile(l_temp, [row_size])
				l_sum = tf.reshape(l_tile, [row_size, emb_size])
				l_update = tf.assign(l_v, tf.subtract(l_sum, mu_v_0))

				sig = tf.nn.relu(tf.matmul(l_v, P))
				
				mu_update = tf.assign(mu_v_1, tf.nn.tanh( tf.matmul(x, W_1) + sig))
				mu_old_value_modify = tf.assign(mu_v_0, mu_v_1)
			
			#summation across column	
			mu_summ = tf.reduce_sum(mu_v_1, axis=0)
			g_embedding = tf.matmul(tf.reshape(mu_summ,[1,10]), W_2)
			#print(g_embedding)
			
			return g_embedding
		
	
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
	

	def l2_norm(self, t, eps=1e-12):
		return tf.sqrt(tf.reduce_sum(tf.square(t),1) + eps)

	#siamese cosine loss
	#math
	#[\frac{l \cdot r}{l2_norm(l) \cdot l2_norm(right)}]
	def siamese_cosine_loss(self, left, right, y):
		#cast true value to float type
		#y = 2 * tf.cast(y, tf.float32) - 1
		#predict value from left and right tensors using cosine formula
		#pred = tf.reduce_sum( tf.multiply(left, right) , 1)/ (self.l2_norm(left) * self.l2_norm(right) + 1e-10)

		#print(tf.nn.l2_loss(y-pred)/ tf.cast(tf.shape(left)[0], tf.float32))
		#return tf.nn.l2_loss(y-pred)/ tf.cast(tf.shape(left)[0], tf.float32)
		#return tf.nn.l2_loss(y-pred)
		return right

	#create model
	def __init__(self, row_size, vector_size, emb_size):
		#input vector/acfg's
		
		#self.y_ = tf.placeholder(tf.float32, [row_size,emb_size], name="input_y")
		with tf.variable_scope("siamese") as siam_scope:
			with tf.variable_scope("acfgs-siamese") as input_scope:
				self.x1 = tf.placeholder(tf.float32, [row_size, vector_size], name="input_x1")
				self.x2 = tf.placeholder(tf.float32, [row_size, vector_size], name="input_x2")
				self.y = tf.placeholder(tf.int32, name="input_y")
			#creating nn using inputs
			self.e1 = self.emb_generation(self.x1, row_size, vector_size, emb_size, "left_embedding")
			#print("-->siamese left tensor", self.e1)
			siam_scope.reuse_variables()
			self.e2 = self.emb_generation(self.x2, row_size, vector_size, emb_size, "right_embedding")
			#print("-->siamese right tensor", self.e2)
		with tf.name_scope("loss"):
			self.loss = self.siamese_cosine_loss(self.e1, self.e2, self.y)
		#create loss
		#self.loss = self.loss_with_spring()