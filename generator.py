'''
	Generator definition for AdvGAN

	ref: https://arxiv.org/pdf/1801.02610.pdf
'''

import tensorflow as tf

# helper function for convolution -> instance norm -> relu
def ConvInstNormRelu(x, filters, kernel_size=3, strides=1):
	Conv = tf.layers.conv2d(
						inputs=x,
						filters=filters,
						kernel_size=kernel_size,
						strides=strides,
						padding="same",
						activation=None)

	InstNorm = tf.contrib.layers.instance_norm(Conv)

	return tf.nn.relu(InstNorm)


# helper function for trans convolution -> instance norm -> relu
def TransConvInstNormRelu(x, filters, kernel_size=3, strides=2):
	TransConv = tf.layers.conv2d_transpose(
						inputs=x,
						filters=filters,
						kernel_size=kernel_size,
						strides=strides,
						padding="same",
						activation=None)

	InstNorm = tf.contrib.layers.instance_norm(TransConv)

	return tf.nn.relu(InstNorm)

# helper function for residual block of 2 convolutions with same num filters
# in the same style as ConvInstNormRelu
def ResBlock(x, filters=32, kernel_size=3, strides=1):
	b1 = ConvInstNormRelu(x, filters=filters, kernel_size=kernel_size, strides=strides)
	b2 = ConvInstNormRelu(b2, filters=filters, kernel_size=kernel_size, strides=strides)

	return x + b2


def generator(x):
	with tf.variable_scope('Generator'):
		input_layer = tf.reshape(x, [-1, 28, 28, 1])

		# define first three conv + inst + relu layers
		l1 = ConvInstNormRelu(input_layer, filters=8, kernel_size=3, strides=1)
		l2 = ConvInstNormRelu(l1, filters=16, kernel_size=3, strides=2)
		l3 = ConvInstNormRelu(l2, filters=32, kernel_size=3, strides=2)

		# define residual blocks
		rb1 = ResBlock(l3, filters=32)
		rb2 = ResBlock(rb1, filters=32)
		rb3 = ResBlock(rb2, filters=32)
		rb4 = ResBlock(rb3, filters=32)

		# upsample using conv transpose
		u1 = TransConvInstNormRelu(rb4, filters=16, kernel_size=3, strides=2)
		u2 = TransConvInstNormRelu(u1, filters=8, kernel_size=3, strides=2)

		# final layer block
		out = ConvInstNormRelu(u2, filters=1, kernel_size=3, strides=1)

		return out



















