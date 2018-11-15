'''
	AdvGAN architecture

	ref: https://arxiv.org/pdf/1801.02610.pdf
'''


import tensorflow as tf
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data

from generator import generator
from discriminator import discriminator
from target_models import Target as target_model



def mse_loss(preds, labels):
	return tf.reduce_mean(tf.square(preds - labels))

def adv_loss(preds, labels):
	real = tf.reduce_sum((labels) * preds, 1)
	other = tf.reduce_max((1 - labels) * preds - (labels * 10000), 1)
	return tf.reduce_sum(tf.maximum(0.0, other - real + .01))

def hinge_loss(preds, labels, c):
	preds = tf.cast(preds, tf.float32)
	labels = tf.cast(labels, tf.float32)
	return tf.reduce_sum(tf.maximum(0.0,tf.abs(preds-tf.tanh(labels)/2)-c))




def AdvGAN(X, y, batch_size=128):
	x_real_pl = tf.placeholder(tf.float32, [None, 28, 28, 1]) # image placeholder
	x_fake_pl = tf.placeholder(tf.float32, [None, 28, 28, 1]) # image placeholder
	d_labels_pl = tf.placeholder(tf.float32, [None, 1])
	y_hinge_pl = tf.placeholder(tf.float32, [None, 28, 28, 1])
	t = tf.placeholder(tf.float32, [None, 10]) # target placeholder


	#-----------------------------------------------------------------------------------
	# MODEL DEFINITIONS

	# gather target model
	f = target_model()

	# generate perturbation, add to original input image(s)
	perturb = generator(x_fake_pl)
	x_perturbed = x_fake_pl + perturb

	disc_batch_x = tf.concat([x_real_pl, x_perturbed], axis=0)

	# pass perturbed image to discriminator and the target model
	d_out_logits, d_out_probs = discriminator(disc_batch_x)
	d_perturb_logits, d_perturb_probs = discriminator(x_perturbed)

	f_out_logits, f_out_probs = f.ModelC(x_perturbed)

	
	# generate labels for discriminator
	# smooth = 0.0
	# d_labels_real = tf.ones_like(d_real_logits) * (1 - smooth)
	# d_labels_fake = tf.zeros_like(d_perturb_logits)

	#-----------------------------------------------------------------------------------
	# LOSS DEFINITIONS
	d_loss = mse_loss(d_out_probs, d_labels_pl)

	l_adv = adv_loss(f_out_probs, t)

	l_hinge = hinge_loss(perturb, y_hinge_pl, 0.3)

	alpha = 1
	beta = 1
	g_loss = mse_loss(d_perturb_probs, d_labels_pl) + alpha*l_adv + beta*l_hinge 

	# ----------------------------------------------------------------------------------
	# gather variables for training/restoring
	t_vars = tf.trainable_variables()
	f_vars = [var for var in t_vars if 'ModelC' in var.name]
	d_vars = [var for var in t_vars if 'd_' in var.name]
	g_vars = [var for var in t_vars if 'g_' in var.name]

	d_opt = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
	g_opt = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)

	saver = tf.train.Saver(f_vars)

	g_saver = tf.train.Saver(g_vars)

	init  = tf.global_variables_initializer()

	sess  = tf.Session()
	sess.run(init)

	saver.restore(sess, "./weights/target_model/model.ckpt")

	for i in range(5000):
		# ------------------------------------------------------------------------------
		# train the discriminator first on real and generated images
		real_image_inp = X[np.random.randint(0, X.shape[0], size=int(batch_size / 2)),:,:,:]
		fake_image_inp = X[np.random.randint(0, X.shape[0], size=int(batch_size / 2)),:,:,:]

		disc_batch_y = np.zeros([batch_size, 1])
		disc_batch_y[0:int(batch_size / 2)] = 1

		_, dl = sess.run([d_opt, d_loss], feed_dict={x_real_pl: real_image_inp, \
													 x_fake_pl: fake_image_inp, \
													 d_labels_pl: disc_batch_y})

		if i % 10 == 0:
			print('discriminator loss: ' + str(dl))

		# train the generator 5x (test)
		for _ in range(5):
			# ------------------------------------------------------------------------------
			# train the generator for perturbed images using loss for discriminator, adversarial, and hinge
			random_samples = np.random.randint(0, X.shape[0], size=int(batch_size))
			fake_image_inp = X[random_samples,...]
			y_discrim = np.ones([batch_size,1])
			target_class = y[random_samples]

			_, gl = sess.run([g_opt, g_loss], feed_dict={x_fake_pl: fake_image_inp, \
														 d_labels_pl: y_discrim, \
														 y_hinge_pl: np.zeros((batch_size, 28, 28, 1)), \
														 t: target_class})
		if i % 10 == 0:
			print('generator loss: ' + str(gl))

	saver.save(sess, "weights/generator/gen.ckpt")


def attack(X, y):
	x_pl = tf.placeholder(tf.float32, [None, 28, 28, 1]) # image placeholder

	perturb = generator(x_pl)

	x_perturbed = x_pl + perturb

	f = target_model()
	f_real_logits, f_real_probs = f.ModelC(x_pl)
	f_fake_logits, f_fake_probs = f.ModelC(x_perturbed)

	t_vars = tf.trainable_variables()
	f_vars = [var for var in t_vars if 'ModelC' in var.name]
	g_vars = [var for var in t_vars if 'g_' in var.name]

	init  = tf.global_variables_initializer()

	sess = tf.Session()
	sess.run(init)

	f_saver = tf.train.Saver(f_vars)
	g_saver = tf.train.Saver(g_vars)
	# f_saver.restore(sess, "./weights/target_model/model.ckpt")
	g_saver.restore(sess, "./weights/generator/gen.ckpt")

	# p, xp, real_l, fake_l = sess.run([perturb, x_perturbed, f_real_probs, f_fake_probs], \
									# feed_dict={x_pl: X})
	real_l = sess.run(x_perturbed, \
									feed_dict={x_pl: X})
	# print(np.argmax(y, axis=1))
	print(real_l.shape)
	# print(np.argmax(fake_l, axis=1))

	# print(p.shape)
	# plt.imshow(p[1,:,:],cmap="Greys_r")





(X,y), (_,_) = mnist.load_data()
X = np.divide(X, 255.0)
X = X.reshape(X.shape[0], 28, 28, 1)
y = to_categorical(y, num_classes=10)

# AdvGAN(X, y, batch_size=128)
rs = np.random.randint(0, X.shape[0], 8)
attack(X[rs,...], y[rs,...])





