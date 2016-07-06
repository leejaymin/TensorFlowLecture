'''
Created on Nov 17, 2015

@author: root
'''
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print sess.run(hello)

a = tf.constant(10)
b = tf.constant(32)
print sess.run(a+b)