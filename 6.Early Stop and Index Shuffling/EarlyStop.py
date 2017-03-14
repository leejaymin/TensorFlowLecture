# Early Stop

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.base import load_iris
import numpy as np

tf.reset_default_graph()


def MLP_iris():
    # load the iris data.
    iris = load_iris()

    np.random.seed(0)
    random_index = np.random.permutation(150)

    iris_data = iris.data[random_index]
    iris_target = iris.target[random_index]
    iris_target_onehot = np.zeros((150, 3))
    iris_target_onehot[np.arange(150), iris_target] = 1

    accuracy_list = []

    # build computation graph
    x = tf.placeholder("float", shape=[None, 4], name='x')
    y_target = tf.placeholder("float", shape=[None, 3], name='y_target')

    W1 = tf.Variable(tf.zeros([4, 128]), name='W1')
    b1 = tf.Variable(tf.zeros([128]), name='b1')
    h1 = tf.sigmoid(tf.matmul(x, W1) + b1, name='h1')

    W2 = tf.Variable(tf.zeros([128, 3]), name='W2')
    b2 = tf.Variable(tf.zeros([3]), name='b2')
    y = tf.nn.softmax(tf.matmul(h1, W2) + b2, name='y')

    cross_entropy = -tf.reduce_sum(y_target * tf.log(y), name='cross_entropy')

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_target, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())

    for i in range(500):
        sess.run(train_step, feed_dict={x: iris_data[0:100], y_target: iris_target_onehot[0:100]})

        train_accuracy = sess.run(accuracy, feed_dict={x: iris_data[0:100], y_target: iris_target_onehot[0:100]})
        validation_accuracy = sess.run(accuracy, feed_dict={x: iris_data[100:], y_target: iris_target_onehot[100:]})
        print (
        "step %d, training accuracy: %.3f / validation accuracy: %.3f" % (i, train_accuracy, validation_accuracy))

        accuracy_list.append(validation_accuracy)

        if i >= 50:
            if validation_accuracy - np.mean(accuracy_list[int(round(len(accuracy_list) / 2)):]) <= 0.01:
                break

    sess.close()


MLP_iris()