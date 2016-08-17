import tensorflow as tf

import input_data


def xaver_init(n_inputs, n_outputs, uniform = True):
    if uniform:
        init_range = tf.sqrt(6.0/ (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)

    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

learning_rate = 0.01
training_epochs = 15
batch_size = 100
display_step = 1

mnist = input_data.read_data_sets("./MNIST_DATA", one_hot=True)
# tensorflow graph input
X = tf.placeholder('float', [None, 784]) # mnist data image of shape 28 * 28 = 784
Y = tf.placeholder('float', [None, 10]) # 0-9 digits recognition = > 10 classes

# set model weights
W1 = tf.get_variable("W1", shape=[784, 256], initializer=xaver_init(784, 256))
W2 = tf.get_variable("W2", shape=[256, 256], initializer=xaver_init(256, 256))
W3 = tf.get_variable("W3", shape=[256, 10], initializer=xaver_init(256, 10))

#W1 = tf.Variable(tf.random_normal([784, 256]))
#W2 = tf.Variable(tf.random_normal([256, 256]))
#W3 = tf.Variable(tf.random_normal([256, 10]))

B1 = tf.Variable(tf.random_normal([256]))
B2 = tf.Variable(tf.random_normal([256]))
B3 = tf.Variable(tf.random_normal([10]))

# Construct model
L1 = tf.nn.relu(tf.add(tf.matmul(X,W1),B1))
L2 = tf.nn.relu(tf.add(tf.matmul(L1,W2),B2)) # Hidden layer with RELU activation
hypothesis = tf.add(tf.matmul(L2, W3), B3) # No need to use softmax here

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, Y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph,
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        # Fit the line.
        for step in xrange(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # Fit training using batch data

            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})

            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' %(epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))