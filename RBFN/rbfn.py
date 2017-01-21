from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

def run_RBFN_MNIST():

	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	
	data = np.concatenate((mnist.train.images, mnist.train.labels), axis=1)
	np.random.shuffle(data)

	x_train = data[:, :mnist.train.images.shape[1]]
	y_train = data[:, mnist.train.images.shape[1]:]

	hidden_layer_size = 100

	y_contig = np.ascontiguousarray(y_train).view(np.dtype((np.void, y_train.dtype.itemsize * y_train.shape[1])))
	n_classes = np.unique(y_contig).view(y_train.dtype).reshape(-1, y_train.shape[1]).shape[0]

	n_examples, n_input = x_train.shape
	n_centers = hidden_layer_size

	sigma = 10 
	beta = 1 / (2 * sigma ** 2)

	x = tf.placeholder(tf.float32, [None, n_input])
	y = tf.placeholder(tf.float32, [None, n_classes])

	c = tf.constant(x_train[:hidden_layer_size, :], dtype=tf.float32)

	x_x = tf.matmul(inner(x,x), tf.ones([1,n_centers]))
	c_c = tf.matmul(tf.ones([n_examples, 1]), tf.transpose(inner(c,c)))
	x_c = tf.matmul(x, tf.transpose(c))
	dist_sqrd = x_x + c_c - 2 * x_c
	rbf_layer = tf.exp(-1 * beta * dist_sqrd)


	weight = tf.Variable(tf.random_normal([n_centers, n_classes]))
	bias = tf.Variable(tf.random_normal([n_classes]))
	out_layer = tf.matmul(rbf_layer, weight) + bias

	cost = cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out_layer))
	optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(out_layer, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

def inner(a, b):
	return tf.reduce_sum(tf.mul(a,b),1,keep_dims=True)
	
run_RBFN_MNIST()






