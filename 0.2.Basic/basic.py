'''
Created on Nov 17, 2015

@author: root
'''

import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# Define some operations
add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print ("Addition with variables: %i" % sess.run(add, feed_dict={a:2, b:3}))
    print ("Multiplication with variables: %d" % sess.run(mul, feed_dict={a:2, b:3}))



