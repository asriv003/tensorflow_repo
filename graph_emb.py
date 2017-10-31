import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf


#To initialize weights and biases
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#input acfg
#x = tf.placeholder(tf.float32, [None, 8])
#for now use randomly generated matrix
#r= 5, m = 8(feature vector size)
x = weight_variable([5,8])

#embeddings to be calculated(rxp)
#r - number of row or acfg size, p - desired dimension of embedding
#assuming p = 10
mu_t_0 = tf.Variable(tf.zeros([5, 10], dtype=tf.float32))
mu_t_1 = tf.Variable(tf.zeros([5, 10], dtype=tf.float32))

#summation of embedding (rxp)
l = tf.Variable(tf.zeros([5, 10], dtype=tf.float32))

#Weights
#W_1 is dxp matrix where d is dimension of input vector and p is desired embedding size
W_1 = weight_variable([8, 10])
#W_2 is pxp matrix to transfrom embedding vector
W_2 = weight_variable([10, 10])

#TODO:
#P matrixs
#it will be nxpxp size since we have to make n layers of pxp matrices
#Currently one layer
P = weight_variable([10, 10])

#TODO: create function for calculating 'l' given mu_t_0
l_temp = tf.reduce_sum(mu_t_0, axis=0)	#column wise summation
l_tile = tf.tile(l_temp, [5])
l_sum = tf.reshape(l_tile, [5,10])

l_update = tf.assign(l, tf.subtract(l_sum, mu_t_0))

#TODO: create a function which takes 'l' as input and return sigma value
#sigma function of ReLu operations
#5x10 = (5x10 * 10x 10)
sig = tf.nn.relu(tf.matmul(l, P))

mu_update = tf.assign(mu_t_1, tf.nn.tanh( tf.matmul(x, W_1) + sig))
mu_old_value_modify = tf.assign(mu_t_0, mu_t_1)

mu_summ = tf.reduce_sum(mu_t_1,axis=0)
graph_embedding = tf.matmul(tf.reshape(mu_summ,[1,10]), W_2)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	
	#for t=1 to T
	for i in range(20):
		#l_v = sum of neighbour mu (embedding values)
		sess.run(l_update)
		#update mu values
		sess.run(mu_update)
		sess.run(mu_old_value_modify)

	#sess.run(graph_embedding)
	print(sess.run(graph_embedding))

