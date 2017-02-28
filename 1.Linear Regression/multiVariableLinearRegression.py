import tensorflow as tf
import numpy as np

xy = np.loadtxt('multiFeaturesTrain.txt', unpack=True, dtype='float32')
x_data = xy[0:-1]
y_data = xy[-1]

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1, 1))
b = tf.Variable(tf.random_uniform([1], -1, 1))

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.001)  # learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)  # goal is minimize cost

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

# Fit the line.
for step in xrange(2001):
    sess.run(train)
    if step % 400 == 0:
        print step, "cost=", "{:.9f}".format(sess.run(cost)), sess.run(W), sess.run(b)

# Learns best fit is W: [1], b[0]

#print sess.run(hypothesis, feed_dict={X: 5})
#print sess.run(hypothesis, feed_dict={X: 2.5})