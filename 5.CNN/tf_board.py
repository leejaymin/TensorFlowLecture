import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time

batch_size = 125
test_size = 200
training_epochs = 20

def init_weights(shape,tag):
    #return tf.Variable(tf.random_normal(shape, stddev=0.01))
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1),name=tag)

# Filter weight vectors 또는 kernel: w, w2, w3, w4, w_0
def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):
        
    l1a = tf.nn.relu(tf.nn.conv2d(X, w,                       # l1a shape=(?, 28, 28, 32)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],              # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)


    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2,                     # l2a shape=(?, 14, 14, 64)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],              # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)


    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3,                     # l3a shape=(?, 7, 7, 128)
                        strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],              # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l3 = tf.nn.dropout(l3, p_keep_conv)


    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o, name="y")
    return pyx

# Read data
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)
#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

# trx.reshape( n-inputs, image size, image size, depth )
 # this variable is input in model()
#trX = trX.reshape(-1, 28, 28, 1)  # 28x28x1 input img
#teX = teX.reshape(-1, 28, 28, 1)  # 28x28x1 input img

#X = tf.placeholder("float", [None, 28, 28, 1], name = 'X')
X = tf.placeholder("float", shape=[None, 784], name = 'x') # none represents variable length of dimension. 784 is the dimension of MNIST data.
Y = tf.placeholder("float", [None, 10], name = 'y')

# reshape input data
x_image = tf.reshape(X, [-1,28,28,1], name="x_image")

w = init_weights([3, 3, 1, 32],"W_conv1")       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64],"W_conv2")     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128],"W_conv3")    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625],"FC_1") # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, 10],"FC_1")         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(x_image, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y),name="cross_entropy")
train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)


# create summary of parameters
tf.summary.histogram('W_conv1', w)
tf.summary.histogram('W_conv2', w2)
tf.summary.histogram('y', py_x)
tf.summary.scalar('cross_entropy', cost)
# init
init = tf.global_variables_initializer()
    
# Launch the graph in a session
with tf.Session() as sess:
    #tensorboard
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter("/tmp/mnistCNN", sess.graph)
    
    # you need to initialize all variables
    start_time = time.time()
    sess.run(init)
    
    for i in range(training_epochs):
        avg_cost = 0.
        avg_training_accuracy = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
              
        for step in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #batch_xs_image = batch_xs.reshape(-1, 28, 28, 1)
                     
            sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys,
                                          p_keep_conv: 0.8, p_keep_hidden: 0.5})
            
            # Training average cost 계산
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, p_keep_conv:1.0, p_keep_hidden:1.0})/total_batch
                        
            avg_training_accuracy += (np.mean(np.argmax(batch_ys, axis=1) ==
                         sess.run(predict_op, feed_dict={X: batch_xs,
                                                        Y: batch_ys,
                                                        p_keep_conv: 1.0,
                                                        p_keep_hidden: 1.0})))/total_batch

            
        print("Epoch: %d, training error: %.4f, training accuracy: %.4f"%(i,avg_cost,avg_training_accuracy))
        # tensorboard를 위해서 기록한다.
        summary = sess.run(merged, feed_dict={X: batch_xs, Y: batch_ys, p_keep_conv:1.0, p_keep_hidden:1.0})
        summary_writer.add_summary(summary , i)

        
        # testing accuracy 계산
        # 인덱스를 뒤 썩어 준다. 랜덤하게 200개 추출을 위해서
        test_indices = np.arange(mnist.test.labels.shape[0]) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size] # 200개만 선택한다. 
        
        #teX = mnist.test.images[test_indices].reshape(-1, 28, 28, 1) # input을 2차원 image를 담은 3차원 matrix로 표현 
        teX = mnist.test.images[test_indices]
        testing_accuracy = np.mean(np.argmax(mnist.test.labels[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX,
                                                        Y: mnist.test.labels[test_indices],
                                                        p_keep_conv: 1.0,
                                                        p_keep_hidden: 1.0}))
        print("Testing Accuracy: %.4f"%(testing_accuracy))
        
        # shuffled testing data 200개에 대해서 accuracy 1.0에 도달하면 Training을 멈춘다.
        if testing_accuracy == 1.0:
            print("Early stop..")
            break
        
    print("time elapsed: {:.2f}s".format(time.time() - start_time))
