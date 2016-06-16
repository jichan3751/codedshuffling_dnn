# from tensorflow website: https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html

# data download & import
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import numpy as np
import time

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						strides=[1, 2, 2, 1], padding='SAME')

def shuffle_dataset(full_data_x, full_data_y, num_examples):
    perm1 = np.random.permutation(num_examples)
    full_data_x_shuffled = full_data_x[perm1,:]
    full_data_y_shuffled = full_data_y[perm1,:]
    return full_data_x_shuffled, full_data_y_shuffled

def data_part(full_data_x, full_data_y,num_examples,n_workers,ind_worker):
    num_data_per_worker = num_examples/n_workers
    data_part_x = full_data_x[(num_data_per_worker)*ind_worker:(num_data_per_worker)*(ind_worker+1)]
    data_part_y = full_data_y[(num_data_per_worker)*ind_worker:(num_data_per_worker)*(ind_worker+1)]
    return data_part_x, data_part_y

def update_averaged_parameter(sess,W_list,n_workers):
    W = sess.run(W_list[0])
    for i in range(1,n_workers):
        W = W + sess.run(W_list[i])
    W = W / n_workers
    for i in range(n_workers):
        assign_op = W_list[i].assign(W)
        sess.run(assign_op)


def train_model(
    x, y_, keep_prob,
    W_conv1, b_conv1,
    W_conv2, b_conv2,
    W_fc1, b_fc1,
    W_fc2, b_fc2,
    step_size
    ):
    
    # 1st conv layer
    x_image = tf.reshape(x, [-1,28,28,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 2nd conv layer

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # fully(densely) connected layer

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout?
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # readout (output)
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # optimizer
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    return train_step, y_conv


# configuration
n_epochs = 20000
n_workers = 20
num_examples = 55000
step_size = 1e-4
batch_size = 50
SHUFFLE_DATA = False

assert (num_examples % (batch_size * n_workers)==0), \
    "(num_examples % (batch_size * n_workers)==0)"


x_list = []
yy_list = [] #y_ans = y_
keep_prob_list = []

W_conv1_list = []
b_conv1_list = []
W_conv2_list = []
b_conv2_list = []
W_fc1_list = []
b_fc1_list = []
W_fc2_list = []
b_fc2_list = []

train_step_list = []
y_conv_list = []
variables_list = [
    W_conv1_list, b_conv1_list, 
    W_conv2_list, b_conv2_list, 
    W_fc1_list, b_fc1_list,
    W_fc2_list,b_fc2_list
    ]

full_data_x, full_data_y = mnist.train.next_batch(num_examples)



start_time = time.time();
# session is for running the operation
sess = tf.InteractiveSession()


# declare variables
for w_i in range(n_workers):
    # inputs
    x = tf.placeholder(tf.float32, [None, 784]) 
    y_ = tf.placeholder(tf.float32, [None, 10]) # answer data
    keep_prob = tf.placeholder(tf.float32)

    # variables in each layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    x_list.append(x)
    yy_list.append(y_)
    keep_prob_list.append(keep_prob)
    W_conv1_list.append(W_conv1)
    b_conv1_list.append(b_conv1)
    W_conv2_list.append(W_conv2)
    b_conv2_list.append(b_conv2)
    W_fc1_list.append(W_fc1)
    b_fc1_list.append(b_fc1)
    W_fc2_list.append(W_fc2)
    b_fc2_list.append(b_fc2)

for i in range(n_workers):
    tmp_train_model, tmp_y_conv = \
        train_model(
            x_list[i], yy_list[i], keep_prob_list[i],
            W_conv1_list[i],b_conv1_list[i],
            W_conv2_list[i],b_conv2_list[i],
            W_fc1_list[i], b_fc1_list[i],
            W_fc2_list[i],b_fc2_list[i],
            step_size
            )

    train_step_list.append(tmp_train_model)
    y_conv_list.append(tmp_y_conv)


# evaluation (only model # 0)
correct_prediction = tf.equal(tf.argmax(y_conv_list[0],1), tf.argmax(yy_list[0],1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



print "shuffle_data", SHUFFLE_DATA
print "n_workers", n_workers


sess.run(tf.initialize_all_variables())

for i in range(n_epochs):
    if SHUFFLE_DATA:
        full_data_x, full_data_y = shuffle_dataset(full_data_x, full_data_y, num_examples)

    for i2 in range(n_workers):
        data_part_x, data_part_y = data_part(full_data_x, full_data_y, num_examples,n_workers,i2)
        num_of_batch = (data_part_x.shape[0])//batch_size
        for i3 in range(num_of_batch):
            batch = (data_part_x[i3*batch_size:(i3+1)*batch_size,:], data_part_y[i3*batch_size:(i3+1)*batch_size,:])
            if i%100==0 and i3==0 and i2==0 :
                train_accuracy = accuracy.eval(feed_dict={
                    x_list[0]:batch[0], 
                    yy_list[0]:batch[1], 
                    keep_prob_list[0]: 1.0
                    })
                print("step %d, training accuracy %g"%(i, train_accuracy))

            sess.run(
                train_step_list[i2], 
                feed_dict={
                    x_list[i2]: batch[0],
                    yy_list[i2]: batch[1],
                    keep_prob_list[i2]: 0.5
                    }
                )
            if i3%55==0:
                print "Running batch %d / %d, worker %d, step %d " \
                    %(i3, num_of_batch, i2, i)

    for var in variables_list:
        update_averaged_parameter(sess,var,n_workers)


print("test accuracy %g"%accuracy.eval(feed_dict={
	x_list[0]: mnist.test.images, yy_list[0]: mnist.test.labels, keep_prob_list[0]: 1.0}))

print 'Running Time : %.02f' % (time.time() - start_time)























