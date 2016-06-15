# from tensorflow website: https://www.tensorflow.org/versions/r0.9/tutorials/mnist/beginners/index.html

# data download & import
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
# import numpy as np


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

def train_model(x, y_, W, b,step_size):
	# model
	y = tf.nn.softmax(tf.matmul(x, W) + b)
	# cross entropy (loss function)
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	# for run: Gradient Descent 
	train_step = tf.train.GradientDescentOptimizer(step_size).minimize(cross_entropy) # run
	return train_step, y


# configuration
n_epochs = 1000
n_workers = 20
num_examples = 55000
step_size = 0.5
batch_size = 20
shuffle_data = False

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
full_data_x, full_data_y = mnist.train.next_batch(num_examples)


	
x_list = []
yy_list = []
W_list = []
b_list = []
train_step_list = []
y_list = []

for i in range(n_workers):
	# inputs
	x_list.append( tf.placeholder(tf.float32, [None, 784]) )
	yy_list.append( tf.placeholder(tf.float32, [None, 10]) ) # answer data
	W_list.append( tf.Variable(tf.zeros([784, 10])) )
	b_list.append( tf.Variable(tf.zeros([10])) )

for i in range(n_workers):
	tmp_train_model, tmp_y = train_model(
		x_list[i], yy_list[i],
		W_list[i],b_list[i],
		step_size
		)
	train_step_list.append(tmp_train_model)
	y_list.append(tmp_y)

# for evaluation: only on node zero
correct_prediction = tf.equal(tf.argmax(y_list[0],1), tf.argmax(yy_list[0],1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# for run: initializing the variables
init = tf.initialize_all_variables() # run

# session is for running the operation
sess = tf.Session()

# run initialization, run GD for 1000 time
sess.run(init)

print "shuffle_data", shuffle_data
print "n_workers", n_workers
for i in range(n_epochs):
	if shuffle_data:
		full_data_x, full_data_y = shuffle_dataset(full_data_x, full_data_y, num_examples)
	for i2 in range(n_workers):
		data_part_x, data_part_y = data_part(full_data_x, full_data_y, num_examples,n_workers,i2)
		for i3 in range((data_part_x.shape[0])//batch_size):
			sess.run(
				train_step_list[i2], 
				feed_dict={
					x_list[i2]: data_part_x[i3*batch_size:(i3+1)*batch_size,:],
					yy_list[i2]: data_part_y[i3*batch_size:(i3+1)*batch_size,:]
					}
				)
	update_averaged_parameter(sess,W_list,n_workers)
	update_averaged_parameter(sess,b_list,n_workers)

	if i % 1 == 0 :
		# Evaluation after GD
		acc_tmp = sess.run(accuracy, feed_dict={x_list[0]: mnist.test.images, yy_list[0]: mnist.test.labels})
		print "finished epoch ", i , "accuracy:",acc_tmp



