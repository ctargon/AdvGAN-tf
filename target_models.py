'''
	Target model definitions that adverserial examples are attempting to 'fool'

	ref: https://arxiv.org/pdf/1801.02610.pdf
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from keras.utils import to_categorical
from keras.datasets import mnist





class Target:
	def __init__(self, lr=0.001, epochs=50, n_input=28, n_classes=10, batch_size=16,\
					restore=0):
		self.lr = lr
		self.epochs = epochs
		self.n_input = 28
		self.n_classes = 10
		self.batch_size = batch_size
		self.restore = restore

		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	# randomly shuffle a dataset 
	def shuffle(self, X, Y):
		rands = random.sample(range(X.shape[0]),X.shape[0])
		return X[rands], Y[rands]

	# get the next batch based on x, y, and the iteration (based on batch_size)
	def next_batch(self, X, Y, i, batch_size):
		idx = i * batch_size
		idx_n = i * batch_size + batch_size
		return X[idx:idx_n], Y[idx:idx_n]


	# USAGE:
	# 		- encoder network for vae
	# PARAMS:
	#	x: input data sample
	#	h_hidden: LIST of num. neurons per hidden layer
	def ModelC(self, x):
		with tf.variable_scope('ModelC', reuse=tf.AUTO_REUSE):
			#input_layer = tf.reshape(x, [-1, 28, 28, 1])

			conv1 = tf.layers.conv2d(
								inputs=x,
								filters=32,
								kernel_size=3,
								padding="same",
								activation=tf.nn.relu)
			
			conv2 = tf.layers.conv2d(
								inputs=conv1,
								filters=32,
								kernel_size=3,
								padding="same",
								activation=tf.nn.relu)

			pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

			conv3 = tf.layers.conv2d(
								inputs=pool1,
								filters=64,
								kernel_size=3,
								padding="same",
								activation=tf.nn.relu)

			conv4 = tf.layers.conv2d(
								inputs=conv3,
								filters=64,
								kernel_size=3,
								padding="same",
								activation=tf.nn.relu)

			pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

			pool2_flatten = tf.contrib.layers.flatten(pool2)

			fc1 = tf.layers.dense(inputs=pool2_flatten, units=200, activation=tf.nn.relu)

			fc2 = tf.layers.dense(inputs=fc1, units=200, activation=tf.nn.relu)

			logits = tf.layers.dense(inputs=fc2, units=self.n_classes, activation=None)

			probs = tf.nn.softmax(logits)

			return logits, probs



	def train(self, X, Y, X_test, Y_test):
		# define placeholders for input data
		x = tf.placeholder(tf.float32, [None, X.shape[1], X.shape[2], X.shape[3]])
		y = tf.placeholder(tf.float32, [None, self.n_classes])

		# define compute graph
		logits, _ = self.ModelC(x)

		# define cost
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

		# optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(cost)

		saver = tf.train.Saver()

		# Initializing the variables
		init = tf.global_variables_initializer()

		sess = tf.Session()
		sess.run(init)

		total_batch = int(X.shape[0] / self.batch_size)

		for epoch in range(1, self.epochs + 1):
			avg_cost = 0.

			for i in range(total_batch):
				batch_x, batch_y = self.next_batch(X, Y, i, self.batch_size)
				
				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

				avg_cost += c / total_batch

			print("Epoch:", '%04d' % (epoch), "cost=", "{:.9f}".format(avg_cost))

		# Test model
		correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

		# Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		accs = []

		total_test_batch = int(X_test.shape[0] / self.batch_size)
		for i in range(total_test_batch):
			batch_x, batch_y = self.next_batch(X_test, Y_test, i, self.batch_size)
			#batch_x = dataset.train.permute(batch_x, idxs)
			accs.append(accuracy.eval({x: batch_x, y: batch_y}, session=sess))

		print('accuracy of test set: {}'.format(sum(accs) / len(accs)))

		saver.save(sess, "./weights/target_model/model.ckpt")
		sess.close() 


from keras.datasets import cifar10

if __name__ == '__main__':
	(X,y), (X_test,y_test) = mnist.load_data()
	X = np.divide(X, 255.0)
	X_test = np.divide(X_test, 255.0)
	X = X.reshape(X.shape[0], 28, 28, 1)
	X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
	y = to_categorical(y, num_classes=10)
	y_test = to_categorical(y_test, num_classes=10)
	cnn = Target()
	cnn.train(X, y, X_test, y_test)



