"""
parallel modification of deep mnist tutorial
from tensorflow website: https://www.tensorflow.org/versions/r0.9/tutorials/mnist/pros/index.html

changelog:

"""
# read data
from tensorflow.examples.tutorials.mnist import input_data

import sys
import tensorflow as tf
import numpy as np
import time


# config
# n_workers = 1
# n_epochs = 20000 # = step
# batch_size = 50


n_workers = 5
n_epochs = 50 #50 = step
batch_size = 50

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

def train_model():
	x = tf.placeholder(tf.float32, [None, 784]) 
	y_ = tf.placeholder(tf.float32, [None, 10])

	x_image = tf.reshape(x, [-1,28,28,1])

	# 1st conv layer
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	# 2nd conv layer
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# fully(densely) connected layer
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# readout (output)
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	# optimizer
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

	# evaluation
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# for assigning variable values (runs out of memory if not used!)
	W_conv1_in = tf.placeholder(tf.float32, [5, 5, 1, 32])
	b_conv1_in = tf.placeholder(tf.float32, [32])
	W_conv2_in = tf.placeholder(tf.float32, [5, 5, 32, 64])
	b_conv2_in = tf.placeholder(tf.float32, [64])
	W_fc1_in =  tf.placeholder(tf.float32, [7 * 7 * 64, 1024])
	b_fc1_in = tf.placeholder(tf.float32, [1024])
	W_fc2_in =  tf.placeholder(tf.float32, [1024, 10])
	b_fc2_in = tf.placeholder(tf.float32, [10])

	W_conv1_assign = W_conv1.assign(W_conv1_in)
	b_conv1_assign = b_conv1.assign(b_conv1_in)
	W_conv2_assign = W_conv2.assign(W_conv2_in)
	b_conv2_assign = b_conv2.assign(b_conv2_in)
	W_fc1_assign = W_fc1.assign(W_fc1_in)
	b_fc1_assign = b_fc1.assign(b_fc1_in)
	W_fc2_assign = W_fc2.assign(W_fc2_in)
	b_fc2_assign = b_fc2.assign(b_fc2_in)

	model_dict = {
		"x": x, "y_": y_, 
		"W_conv1": W_conv1, "b_conv1":b_conv1, 
		"W_conv2": W_conv2, "b_conv2":b_conv2, 
		"W_fc1": W_fc1, "b_fc1":b_fc1, 
		"W_fc2": W_fc2, "b_fc2":b_fc2,
		"W_conv1_in": W_conv1_in, "b_conv1_in":b_conv1_in, 
		"W_conv2_in": W_conv2_in, "b_conv2_in":b_conv2_in, 
		"W_fc1_in": W_fc1_in, "b_fc1_in":b_fc1_in, 
		"W_fc2_in": W_fc2_in, "b_fc2_in":b_fc2_in,
		"W_conv1_assign": W_conv1_assign, "b_conv1_assign":b_conv1_assign, 
		"W_conv2_assign": W_conv2_assign, "b_conv2_assign":b_conv2_assign, 
		"W_fc1_assign": W_fc1_assign, "b_fc1_assign":b_fc1_assign, 
		"W_fc2_assign": W_fc2_assign, "b_fc2_assign":b_fc2_assign,
		"keep_prob": keep_prob,
		"train_step": train_step,
		"accuracy": accuracy
	}

	return model_dict

def current_batch( ind_worker, ind_batch, full_data_x, full_data_y, data_order, n_workers, batch_size):
	num_data_per_worker = full_data_x.shape[0]/n_workers
	current_data_indices = data_order[(num_data_per_worker)*ind_worker + (batch_size)*ind_batch : (num_data_per_worker)*ind_worker + (batch_size)*(ind_batch+1)]
	batch_x = full_data_x[ current_data_indices , : ]
	batch_y = full_data_y[ current_data_indices , : ]
	return batch_x, batch_y


def update_averaged_parameter(sess,model_dict_list,n_workers):
	var_list = [
		"W_conv1", "b_conv1",
		"W_conv2", "b_conv2",
		"W_fc1", "b_fc1",
		"W_fc2", "b_fc2"
	]
	for var_key in var_list:
		# get averaged value
		var_array = sess.run(model_dict_list[0][var_key])
		for worker in range(1,n_workers):
			var_array = var_array + sess.run(model_dict_list[worker][var_key])
		var_array = var_array / n_workers
		
		# update values in variables
		var_assign_key = var_key+"_assign"
		var_input_key = var_key+"_in"
		for worker in range(n_workers):
			sess.run(model_dict_list[worker][var_assign_key], feed_dict={model_dict_list[worker][var_input_key]: var_array })





#script start
assert (len(sys.argv)==2), "need right number of arguments"
if int(sys.argv[1])==1:
	SHUFFLE_DATA_GLOBAL = True
	SHUFFLE_DATA_INTERNAL = False
	print "Global Shuffling"
elif int(sys.argv[1])==2:
	SHUFFLE_DATA_GLOBAL = False
	SHUFFLE_DATA_INTERNAL = True
	print "Internal Shuffling"
else:
	SHUFFLE_DATA_GLOBAL = False
	SHUFFLE_DATA_INTERNAL = False
	print "no Shuffling"



np.random.seed(0)
tf.set_random_seed(0)


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


num_train = mnist.train.images.shape[0]
data_order = np.array(range(num_train))


assert (num_train % (batch_size * n_workers)==0), \
	"(num_train % (batch_size * n_workers)==0)"
num_batch_per_worker = num_train / (batch_size * n_workers)

# make models for each worker
model_dict_list = []
for i in range(n_workers):
	model_dict_list.append(train_model())


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

start_time = time.time();

print "start training..."
for step in range(n_epochs):
	# print "step %d parameter update" % step
	update_averaged_parameter(sess,model_dict_list,n_workers)
	for ind_worker in range(n_workers):

		if SHUFFLE_DATA_INTERNAL:
			batch_seq = np.random.permutation(num_batch_per_worker)
		else:
			batch_seq = range(num_batch_per_worker)

		for ind_batch in batch_seq:
			batch_xs, batch_ys = current_batch(
				ind_worker, ind_batch, 
				mnist.train.images, mnist.train.labels, 
				data_order, n_workers, batch_size
			)

			# if step%100 == 0 and ind_worker==0 :
			# 	train_accuracy = model_dict_list[0]["accuracy"].eval(
			# 		feed_dict={
			# 			model_dict_list[0]["x"]: batch_xs,
			# 			model_dict_list[0]["y_"]: batch_ys, 
			# 			model_dict_list[0]["keep_prob"]: 1.0
			# 		}
			# 	)
			# 	print("step %d, worker %d, batch %d training accuracy %g"%(step, ind_worker, ind_batch, train_accuracy))
			# print "step %d, worker %d, batch %d" % (step, ind_worker, ind_batch)
			model_dict_list[ind_worker]["train_step"].run(
				feed_dict={
					model_dict_list[ind_worker]["x"]: batch_xs,
					model_dict_list[ind_worker]["y_"]: batch_ys, 
					model_dict_list[ind_worker]["keep_prob"]: 0.5
				}
			)

	# print "step %d parameter update" % step
	# update_averaged_parameter(sess,model_dict_list,n_workers)
	print(
		"step %d ,test accuracy %g" % (
			step,
			model_dict_list[0]["accuracy"].eval(
				feed_dict={
					model_dict_list[0]["x"]: mnist.test.images, 
					model_dict_list[0]["y_"]: mnist.test.labels, 
					model_dict_list[0]["keep_prob"]: 1.0
				}
			)
		)	
	)

	if SHUFFLE_DATA_GLOBAL:
		data_order = np.random.permutation(num_train)

print 'Running Time : %.02f sec' % (time.time() - start_time)




