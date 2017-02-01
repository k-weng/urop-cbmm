from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cdist, pdist

def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train = [mnist.train.images, mnist.train.labels]
    test = [mnist.test.images, mnist.test.labels]
    val = [mnist.validation.images, mnist.validation.labels]
    rbf_layer_size = 100
    trainRBFN(train[0], train[1], test[0], test[1], val[0], val[1], rbf_layer_size)
    
def RBF_unit_test():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    sess = tf.InteractiveSession()

    perm = np.arange(mnist.train.num_examples)
    np.random.shuffle(perm)

    x = mnist.train.images[:1000,:]
    y = mnist.train.labels[:1000,:]
    c = mnist.train.images[perm][:100,:]

    dist_c = pdist(c)
    sigma = 2 * np.sum(dist_c) / dist_c.shape[0]
    beta = 1 / (2 * sigma ** 2)

    sq_dist_tf = sq_cdist_tf(x, c)
    sq_dist_tf = tf.exp(-sq_dist_tf * beta).eval()
    sq_dist_tf = np.concatenate([np.ones([1000, 1]), sq_dist_tf], axis=1)
    theta = np.linalg.pinv(np.matmul(np.transpose(sq_dist_tf), sq_dist_tf))
    theta = np.matmul(theta, np.transpose(sq_dist_tf))
    theta = np.matmul(theta, y)

    sq_dist_tf_check = sq_cdist_tf_check(x, np.transpose(c))
    sq_dist_tf_check = tf.exp(sq_dist_tf_check * beta).eval()
    sq_dist_tf_check = np.concatenate([np.ones([1000, 1]), sq_dist_tf_check], axis=1)
    theta_check = np.linalg.pinv(np.matmul(np.transpose(sq_dist_tf_check), sq_dist_tf_check))
    theta_check = np.matmul(theta_check, np.transpose(sq_dist_tf_check))
    theta_check = np.matmul(theta_check, y)

def trainRBFN(x_train, y_train, x_test, y_test, x_val, y_val, n_centers):

    n_features = x_train.shape[1]
    n_classes = y_train.shape[1]

    batch_size = x_train.shape[0] / 10

    x = tf.placeholder(tf.float32, [None, n_features])
    y_ = tf.placeholder(tf.float32, [None, n_classes])

    centers = get_centers_random(x_train, n_centers)
    weights = tf.Variable(tf.truncated_normal([n_centers, n_classes], stddev=0.1))

    # biases = tf.Variable(tf.random_normal([1, n_classes], stddev=0.1))
    biases = tf.Variable(tf.zeros([1, n_classes]))

    y = rbf_model(x, centers, n_centers, weights, biases)

    sess = tf.InteractiveSession()

    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)) + 0.01 * tf.nn.l2_loss(weights)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
    train_step = tf.train.AdamOptimizer(1e-2).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch = next_batch(batch_size, x_train, y_train)

        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})
            print("step %d, training accuracy %g"%(i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
        sess.run([train_step, cost], feed_dict={x: batch[0], y_: batch[1]})

    # print("test accuracy %g"%accuracy.eval(feed_dict={x: x_test, y_: y_test}))
    # print("test accuracy %g"%accuracy.eval(feed_dict={x: x_val, y_: y_val}))
    print("test accuracy %g"%accuracy.eval(feed_dict={x: x_train, y_: y_train}))
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

def next_batch(batch_size, x_train, y_train):

    perm = np.arange(x_train.shape[0])
    np.random.shuffle(perm)

    return (x_train[perm][:batch_size, :], y_train[perm][:batch_size, :])

def get_centers_random(x_train, n_centers):

    perm = np.arange(x_train.shape[0])
    np.random.shuffle(perm)

    centers = x_train[perm][:n_centers, :]

    return centers.astype(np.float32)

def sq_cdist_tf(a, b):

    a_a = tf.reduce_sum(a * a, reduction_indices=1, keep_dims=True)
    b_b = tf.reduce_sum(b * b, reduction_indices=1, keep_dims=False)
    a_b = tf.matmul(a, tf.transpose(b))

    return a_a + b_b - 2 * a_b

def sq_cdist_tf_check(a, b):

    b_b =  tf.reduce_sum(b*b, reduction_indices=0, keep_dims=True)
    a_a =  tf.reduce_sum(a*a, reduction_indices=1, keep_dims=True) 

    return 2.0 * tf.matmul(a,b) - tf.add(b_b, a_a)


def rbf_model(inputs, centers, n_centers, weights, biases):

    pdist_centers = pdist(centers)
    sigma = 2 * np.sum(pdist_centers) / pdist_centers.shape[0]
    beta = 1 / (2 * sigma ** 2)

    rbf_layer = tf.exp(-1 * beta * sq_cdist_tf(inputs, centers))

    y = tf.matmul(rbf_layer, weights) + biases

    return y

if __name__ == '__main__':
  main()