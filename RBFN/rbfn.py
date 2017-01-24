from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist, pdist

def run_RBFN_MNIST_TF():
    """
    Creates a TF session and test the RBFN model using MNIST data points
    """

    # learning_rate = 0.0001
    # training_epochs = 15
    # batch_size = 100
    # display_step = 1

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    hidden_layer_size = 1000

    n_centers = hidden_layer_size
    n_input = 784
    n_classes = 10

    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    perm = np.arange(mnist.train.num_examples)
    np.random.shuffle(perm)
    c = mnist.train.images[perm][:n_centers, :]

    weight = tf.Variable(tf.random_normal([n_centers, n_classes], stddev=0.1))
    bias = tf.Variable(tf.ones([n_classes]))

    output = rbf_model(x, c, weight, bias)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    # cost = cdist_tf(output, y)
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
    # sess.run(optimizer, feed_dict={x: [mnist.train.images[0]], y: [mnist.train.labels[0]]})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # # Initializing the variables
    # init = tf.global_variables_initializer()

    # # Launch the graph
    # with tf.Session() as sess:
    #     sess.run(init)

    #     # Training cycle
    #     for epoch in range(training_epochs):
    #         avg_cost = 0.
    #         total_batch = int(mnist.train.num_examples/batch_size)
    #         # Loop over all batches
    #         for i in range(total_batch):
    #             batch_x, batch_y = mnist.train.next_batch(batch_size)
    #             # Run optimization op (backprop) and cost op (to get loss value)
    #             _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
    #                                                           y: batch_y})
    #             # Compute average loss
    #             avg_cost += c / total_batch
    #         # Display logs per epoch step
    #         if epoch % display_step == 0:
    #             print("Epoch:", '%04d' % (epoch+1), "cost=", \
    #                 "{:.9f}".format(avg_cost))
    #     print("Optimization Finished!")

    #     # Test model
    #     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    #     # Calculate accuracy
    #     accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    #     print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

def inner_tf(a, b):
    """
    Calculates the inner product of two matrices in TensorFlow
    """
    return tf.reduce_sum(tf.mul(a,b),1,keep_dims=True)

def cdist_tf(a, b):
    """
    Computes squared Euclidean distance between each pair of the two collections of inputs in Tensorflow
    """

    a_a = tf.matmul(inner_tf(a,a), tf.ones([1, tf.shape(b)[0]]))
    b_b = tf.matmul(tf.ones([tf.shape(a)[0], 1]), tf.transpose(inner_tf(b,b)))
    a_b = tf.matmul(a, tf.transpose(b))

    return a_a + b_b - 2 * a_b
    
def rbf_model(x, c, weight, bias):
    """
    Creates the RBF network model with hidden layer being consists of prototypes selected from random.
    Creates a weight and bias term for the output layer and uses gradient descent to train the model.
    """
    alpha = 20
    p = 10
    min_dist_c = pdist(c)
    min_dist_c = np.sort(min_dist_c)
    sigma = (alpha / p) * np.sum(c[:p])
    beta = 1 / (2 * sigma ** 2)

    n_examples = tf.shape(x)[0]
    n_centers = c.shape[0]

    # x_x = tf.matmul(inner(x,x), tf.ones([1, n_centers]))
    # c_c = tf.matmul(tf.ones([n_examples, 1]), tf.transpose(inner(c,c)))
    # x_c = tf.matmul(x, tf.transpose(c))

    # dist_sqrd = x_x + c_c - 2 * x_c
    sq_dist_tf = cdist_tf(x, c)
    rbf_layer = tf.exp(-1 * beta * sq_dist_tf)

    output = tf.matmul(rbf_layer, weight) + bias

    return output

def test():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    hidden_layer_size = 100

    n_centers = hidden_layer_size
    n_input = 784
    n_classes = 10
    n_examples = 10

    perm = np.arange(mnist.train.num_examples)
    np.random.shuffle(perm)
    x = mnist.train.images[:n_examples,:]
    y = mnist.train.labels[:n_examples,:]
    c = mnist.train.images[perm][:n_centers,:]

    sess = tf.InteractiveSession()
    sq_dist_np = cdist(x, c, 'sqeuclidean')
    sq_dist_tf = cdist_tf(x, c).eval()
    # x_x = tf.matmul(inner_tf(x,x), tf.ones([1, n_centers])).eval()
    # c_c = tf.matmul(tf.ones([n_examples, 1]), tf.transpose(inner_tf(c,c))).eval()
    # x_c = tf.matmul(x, tf.transpose(c)).eval()
    # sq_dist_tf = x_x + c_c - 2 * x_c
    print np.equal(np.round(sq_dist_np), np.round(sq_dist_tf))

run_RBFN_MNIST_TF()


