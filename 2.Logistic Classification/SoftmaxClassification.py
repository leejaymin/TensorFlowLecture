import tensorflow as tf
import numpy as np

xy = np.loadtxt('softmaxTrain.txt', unpack=True, dtype='float')

x_data = np.transpose(xy[0:3])
y_data = np.transpose(xy[3:])
# tensorflow graph input
X = tf.placeholder('float', [None, 3]) # x1, x2, and 1 (for bias)
Y = tf.placeholder('float', [None, 3]) # A, B, C = > three classes
# set model weights
W = tf.Variable(tf.random_uniform([3,3],-1.0, 1.0))

# Our hypothesis
hypothesis = tf.nn.softmax(tf.matmul(X, W)) # softmax

# Cost function: cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), reduction_indices=1))

# Minimize
a = tf.Variable(0.2) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

# Before starting, initialize the variables. We will `run` this first.
init = tf.initialize_all_variables()

# Launch the graph,
with tf.Session() as sess:
    sess.run(init)

    # Fit the line.
    for step in xrange(10000):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis+0.5), Y)
    # Calculate accuracy (re-substitution error)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis+0.5), correct_prediction, accuracy], feed_dict={X:x_data, Y:y_data})
    print "Accuracy:", accuracy.eval({X:x_data, Y:y_data})

    # Test & one-hot encoding
    a = sess.run(hypothesis, feed_dict={X:[[1, 11, 7]]})
    print a, sess.run(tf.arg_max(a,1))

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
    print b, sess.run(tf.arg_max(b, 1))

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
    print b, sess.run(tf.arg_max(c, 1))

    all = sess.run(hypothesis, feed_dict={X:[[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print all, sess.run(tf.arg_max(all, 1))
