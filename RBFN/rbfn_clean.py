from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn import datasets
from sklearn.cross_validation import train_test_split

def get_iris_data():
    RANDOM_SEED = 42
    tf.set_random_seed(RANDOM_SEED)
    iris   = datasets.load_iris()
    data   = iris["data"]
    target = iris["target"]

    N, M  = data.shape
    all_X = np.ones((N, M + 1))
    all_X[:, 1:] = data

    num_labels = len(np.unique(target))
    all_Y = np.eye(num_labels)[target] 
    return train_test_split(all_X, all_Y, test_size=0.33, random_state=RANDOM_SEED)

def test_Iris_tf():
    train_x, test_x, train_y, test_y = get_iris_data()
    train_y = np.asarray(train_y, dtype=np.float32)
    test_y = np.asarray(test_y, dtype=np.float32)
    print train_x.shape

    n_input_dims = train_x.shape[1]   # Number of input nodes: 4 features and 1 bias
    n_centers = 100              # Number of hidden nodes
    n_classes = train_y.shape[1]   # Number of outcomes (3 iris flowers)

    # print n_input_dims
    # print n_centers
    # print n_classes

    # Symbols
    x = tf.placeholder(tf.float32, shape=[None, n_input_dims], name="x")
    y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="y_")

    perm = np.arange(train_x.shape[0])
    np.random.shuffle(perm)
    c = train_x[perm][:n_centers, :]
    c = c.astype(np.float32)
    # print c.shape

    weights = tf.Variable(tf.truncated_normal([n_centers, n_classes], stddev=0.1))
    # biases = tf.Variable(tf.random_normal([1, n_classes], stddev=0.1))
    biases = tf.Variable(tf.zeros([1, n_classes]))

    y = rbf_model(x, c, n_centers, weights, biases)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)) + 0.01 * tf.nn.l2_loss(weights)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    # train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    for epoch in range(1000):
        # Train with each example
        for i in range(len(train_x)):
            # print train_x[i: i+1]
            _, loss = sess.run([train_step, cost], feed_dict={x: train_x[i: i + 1], y_: train_y[i: i + 1]})
            if epoch % 25 == 0 and i == len(train_x) - 1:
                print loss


        train_accuracy = accuracy.eval(feed_dict={x:train_x, y_: train_y})
        test_accuracy  = accuracy.eval(feed_dict={x:test_x, y_: test_y})

        if epoch % 25 == 0:
            print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
                  % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

def test_MNIST_tf():

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    rbf_layer_size = 1000

    n_input_dims = 784
    n_classes = 10
    n_centers = rbf_layer_size

    x = tf.placeholder(tf.float32, [None, n_input_dims])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    perm = np.arange(mnist.train.num_examples)
    np.random.shuffle(perm)
    c = mnist.train.images[perm][:n_centers, :]
    print c.shape

    weights = tf.Variable(tf.truncated_normal([n_centers, n_classes], stddev=0.1))
    # biases = tf.Variable(tf.random_normal([1, n_classes], stddev=0.1))
    biases = tf.Variable(tf.zeros([1, n_classes]))

    y = rbf_model(x, c, n_centers, weights, biases)

    sess = tf.InteractiveSession()
    # tf.global_variables_initializer().run()

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)) + 0.1 * tf.nn.l2_loss(weights)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    # train_step = tf.train.GradientDescentOptimizer(0.03).minimize(cost)
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())
    # losses = []
    for i in range(10000):
        batch = mnist.train.next_batch(100)
        # print type(batch[0])
        if i % 10 == 0:
            print batch[1].shape

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
            print("step %d, training accuracy %g"%(i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        sess.run([train_step, cost], feed_dict={x: batch[0], y_: batch[1]})
        # print loss
    # print sess.run(weights)

    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.validation.images, y_: mnist.validation.labels}))
    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels}))

def sq_cdist_tf(a, b):

    a_a = tf.reduce_sum(a * a, reduction_indices=1, keep_dims=True)
    b_b = tf.reduce_sum(b * b, reduction_indices=1, keep_dims=False)
    a_b = tf.matmul(a, tf.transpose(b))

    return a_a + b_b - 2 * a_b

def rbf_model(inputs, centers, n_centers, weights, biases):
    pdist_centers = pdist(centers)
    sigma = 2 * np.sum(pdist_centers) / pdist_centers.shape[0]
    beta = 1 / (2 * sigma ** 2)

    rbf_layer = tf.exp(-1 * beta * sq_cdist_tf(inputs, centers))

    y = tf.matmul(rbf_layer, weights) + biases

    return y

test_MNIST_tf()
# test_Iris_tf()