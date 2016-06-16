
"""
1. make graph & initialize

2. assign random W_ans,b_ans to variables

	feed random x and get output y

3. initialize graph again to get variables randomize

4. run optimizer

"""

import tensorflow as tf
import numpy as np
import time


def train_model(n_in, n_hidden, n_out, step_size):

	# inputs
	x = tf.placeholder(tf.float32, [None, n_in]) 
	y_ = tf.placeholder(tf.float32, [None, n_out]) # answer data

	# variables
	W = tf.Variable(tf.random_normal((n_in, n_hidden), 0.0, 1.0))
	b = tf.Variable(tf.zeros([n_hidden]))

	W2 = tf.Variable(tf.random_normal((n_hidden, n_out), 0.0, 1.0)) 
	b2 = tf.Variable(tf.zeros([n_out]))
	
	# model
	h = tf.nn.softmax(tf.matmul(x, W) + b)
	y = tf.nn.softmax(tf.matmul(h, W2) + b2)

	# cross entropy (loss function) for regression
	loss = tf.reduce_mean(tf.square(y - y_))

	# for run: Gradient Descent 
	train_step = tf.train.GradientDescentOptimizer(step_size).minimize(loss) # run

	model_dict = {"x": x, "y_": y_, "W": W, "b":b, "W2":W2, "b2":b2, "y":y, "loss":loss,  "train_step": train_step}
	return model_dict

def generate_dataset(model_dict_list, num_examples, num_test, n_in):
	# generate synthetic data (from model #0)
	x_data_full = np.random.normal(0, 1, (num_examples + num_test, n_in))
	y_data_full = sess.run(model_dict_list[0]["y"], feed_dict={model_dict_list[0]["x"]: x_data_full} )
	var_ans_data = {
		"W": sess.run(model_dict_list[0]["W"]),
		"b": sess.run(model_dict_list[0]["b"]), 
		"W2": sess.run(model_dict_list[0]["W2"]),
		"b2": sess.run(model_dict_list[0]["b2"])
		} 
	x_data = x_data_full[0:num_examples,:]
	y_data = y_data_full[0:num_examples,:]
	x_data_test = x_data_full[num_examples:,:]
	y_data_test = y_data_full[num_examples:,:]
	return var_ans_data, x_data, y_data, x_data_test, y_data_test


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

def update_averaged_parameter(sess,model_dict_list,n_workers):
	# ..
	ignore_list = ["x","y_","y", "loss", "train_step"]
	for var_key in model_dict_list[0]:
		if var_key in ignore_list:
			continue
		
		# get averaged value
		W_tmp = sess.run(model_dict_list[0][var_key])
		for worker in range(1,n_workers):
			W_tmp = W_tmp + sess.run(model_dict_list[worker][var_key])
		W_tmp = W_tmp / n_workers
		
		# update values in vaurables
		for worker in range(n_workers):
			sess.run(model_dict_list[worker][var_key].assign(W_tmp))


#config
n_in = 784
n_hidden = 15
n_out =10
n_epochs = 10
n_workers = 10
num_examples = 55000 # num of train set
num_test = 5000
step_size = 100  #1e-1
batch_size = 50
SHUFFLE_DATA = False
# SHUFFLE_DATA = True

np.random.seed(0)
tf.set_random_seed(0)

assert (num_examples % (batch_size * n_workers)==0), \
	"(num_examples % (batch_size * n_workers)==0)"

print "shuffle data:", SHUFFLE_DATA
print "step size:", step_size
print "n_worker", n_workers
print "batch_size", batch_size

# make train models for each worker
print "making models..."
model_dict_list = []
for i in range(n_workers):
	model_dict_list.append(train_model(n_in, n_hidden, n_out, step_size))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# generate_dataset
print "generating datasets..."
var_ans_data, x_data, y_data, x_data_test, y_data_test = \
	generate_dataset(model_dict_list, num_examples, num_test, n_in)
	# add save & load here

# learning part

sess.run(tf.initialize_all_variables())

start_time = time.time();


print "start training..."
for step in range(n_epochs):
	#check error
	if step%1==0:
		test_error = sess.run(
			model_dict_list[0]["loss"], 
			feed_dict={
				model_dict_list[0]["x"]: x_data_test,
				model_dict_list[0]["y_"]: y_data_test,
				}
			) 
		print("epoch %d, test error %g"%(step, test_error)) # not normalized yet

	if SHUFFLE_DATA:
		x_data, y_data = shuffle_dataset(x_data,y_data,num_examples)
	for worker in range(n_workers):
		data_part_x, data_part_y = data_part(x_data, y_data, num_examples,n_workers,worker)
		num_of_batch = (data_part_x.shape[0])//batch_size
		for batch_index in range(num_of_batch):
			batch_x = data_part_x[batch_index*batch_size:(batch_index+1)*batch_size,:]
			batch_y = data_part_y[batch_index*batch_size:(batch_index+1)*batch_size,:]
			sess.run(
				model_dict_list[worker]["train_step"],
				feed_dict={
					model_dict_list[worker]["x"]: batch_x,
					model_dict_list[worker]["y_"]: batch_y
					}
				)
	update_averaged_parameter(sess,model_dict_list,n_workers)
	

test_error = sess.run(
	model_dict_list[0]["loss"], 
	feed_dict={
		model_dict_list[0]["x"]: x_data_test,
		model_dict_list[0]["y_"]: y_data_test,
		}
	) 
print("   final test error %g"%(test_error)) # not normalized yet

print 'Running Time : %.02f sec' % (time.time() - start_time)
# import ipdb; ipdb.set_trace()