import tensorflow as tf
with tf.device("/cpu:0"):
#with tf.device('/gpu:0')
	# Benchmarkning
	a = tf.zeros(shape=[10000, 1000], dtype=tf.float32)
	b = tf.zeros(shape=[1000, 1000], dtype=tf.float32)
	c = tf.matmul(a, b)

sess = tf.Session()
for i in range(200):
	print i
	print sess.run(c)