import tensorflow as tf

a = tf.constant(2, name="a")
b = tf.constant(3, name="b")

x = tf.add(a,b)

with tf.Session() as sess:
	#To use tensorboard
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	print(sess.run(x))

writer.close()

#Command to run tensorboard
#tensorboard --logdir="./graphs" --port 6006
#In browser - http://localhost:6006/