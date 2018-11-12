'''
	Discriminator definition for AdvGAN

	ref: https://arxiv.org/pdf/1801.02610.pdf
'''

import tensorflow as tf

def discriminator(x):
	with tf.variable_scope('Discriminator'):
		input_layer = tf.reshape(x, [-1, 28, 28, 1])

		conv1 = tf.layers.conv2d(
							inputs=input_layer,
							filters=8,
							kernel_size=4,
							strides=2,
							padding="same",
							activation=tf.nn.leaky_relu(alpha=0.2))

		
		conv2 = tf.layers.conv2d(
							inputs=conv1,
							filters=16,
							kernel_size=4,
							strides=2,
							padding="same",
							activation=tf.nn.leaky_relu(alpha=0.2))

		in1 = tf.contrib.layers.instance_norm(conv2)

		conv3 = tf.layers.conv2d(
							inputs=pool1,
							filters=32,
							kernel_size=4,
							strides=2,
							padding="same",
							activation=tf.nn.leaky_relu(alpha=0.2))

		in2 = tf.contrib.layers.instance_norm(conv3)

		in2_flatten = tf.contrib.layers.flatten(in2)

		logits = tf.layers.dense(inputs=in2_flatten, units=1, activation=None)

		probs = tf.nn.sigmoid(logits)

		return logits, probs






