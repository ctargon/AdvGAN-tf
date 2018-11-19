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
import random

from generator import generator
from discriminator import discriminator
from target_models import Target as target_model

def shuffle(X, Y):
	rands = random.sample(xrange(X.shape[0]),X.shape[0])
	return X[rands], Y[rands]

def next_batch(X, Y, i, batch_size):
	idx = i * batch_size
	idx_n = i * batch_size + batch_size
	return X[idx:idx_n], Y[idx:idx_n]


def mse_loss(preds, labels):
	return tf.reduce_mean(tf.square(preds - labels))

def adv_loss(preds, labels):
	real = tf.reduce_sum(labels * preds, 1)
	other = tf.reduce_max((1 - labels) * preds - (labels * 10000), 1)
	return tf.reduce_sum(tf.maximum(0.0, real - other + .01))

def perturb_loss(preds, labels, c):
	preds = tf.cast(preds, tf.float32)
	labels = tf.cast(labels, tf.float32)
	return tf.reduce_sum(tf.maximum(0.0,tf.abs(preds-tf.tanh(labels)/2)-c))




def AdvGAN(X, y, epochs=50, batch_size=128):
	x_pl = tf.placeholder(tf.float32, [None, 28, 28, 1]) # image placeholder
	y_hinge_pl = tf.placeholder(tf.float32, [None, 28, 28, 1])
	t = tf.placeholder(tf.float32, [None, 10]) # target placeholder


	#-----------------------------------------------------------------------------------
	# MODEL DEFINITIONS

	# gather target model
	f = target_model()

	# generate perturbation, add to original input image(s)
	perturb = generator(x_pl)
	x_perturbed = tf.clip_by_value(perturb, -0.3, 0.3) + x_pl
	x_perturbed = tf.clip_by_value(x_perturbed, 0, 1)

	# pass real and perturbed image to discriminator and the target model
	d_real_logits, d_real_probs = discriminator(x_pl)
	d_fake_logits, d_fake_probs = discriminator(x_perturbed)
	
	# pass real and perturbed images to the model we are trying to fool
	f_real_logits, f_real_probs = f.ModelC(x_pl)
	f_fake_logits, f_fake_probs = f.ModelC(x_perturbed)

	
	# generate labels for discriminator (optionally smooth labels for stability)
	smooth = 0.0
	d_labels_real = tf.ones_like(d_real_probs) * (1 - smooth)
	d_labels_fake = tf.zeros_like(d_fake_probs)

	#-----------------------------------------------------------------------------------
	# LOSS DEFINITIONS
	d_loss_real = tf.losses.mean_squared_error(predictions=d_real_probs, labels=d_labels_real)
	d_loss_fake = tf.losses.mean_squared_error(predictions=d_fake_probs, labels=d_labels_fake)
	d_loss = d_loss_real + d_loss_fake

	g_loss_fake = tf.losses.mean_squared_error(predictions=d_fake_probs, labels=tf.ones_like(d_fake_probs))

	l_perturb = perturb_loss(perturb, y_hinge_pl, 0.3)

	l_adv = adv_loss(f_fake_probs, t)

	alpha = 10
	beta = 1
	g_loss = g_loss_fake + alpha*l_adv + beta*l_perturb 

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
	d_saver = tf.train.Saver(d_vars)

	init  = tf.global_variables_initializer()

	sess  = tf.Session()
	sess.run(init)

	saver.restore(sess, "./weights/target_model/model.ckpt")

	total_batches = X.shape[0] / batch_size

	for epoch in range(epochs):

		X, y = shuffle(X, y)
		loss_D_sum = 0.0
		loss_G_fake_sum = 0.0
		loss_perturb_sum = 0.0
		loss_adv_sum = 0.0

		for i in range(total_batches):

			batch_x, batch_y = next_batch(X, y, i, batch_size)

			# train the discriminator first n times
			for _ in range(1):
				_, loss_D_batch = sess.run([d_opt, d_loss], feed_dict={x_pl: batch_x})

			# train the generator n times
			for _ in range(1):
				_, loss_G_fake_batch, loss_adv_batch, loss_perturb_batch = \
									sess.run([g_opt, g_loss_fake, l_adv, l_perturb], \
												feed_dict={x_pl: batch_x, \
														   y_hinge_pl: np.zeros((batch_size, 28, 28, 1)), \
														   t: batch_y})
		loss_D_sum += loss_D_batch
		loss_G_fake_sum += loss_G_fake_batch
		loss_perturb_sum += loss_perturb_batch
		loss_adv_sum += loss_adv_batch

		print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f, \
				\nloss_perturb: %.3f, loss_adv: %.3f, \n" %
				(epoch, loss_D_sum/total_batches, loss_G_fake_sum/total_batches,
				loss_perturb_sum/total_batches, loss_adv_sum/total_batches))

		print('discriminator loss: ' + str(dl))
		# pert, fake_l, real_l = sess.run([x_perturbed, f_out_probs, f_real_probs], feed_dict={x_fake_pl: fake_image_inp})
		# print('LA: ' + str(np.argmax(target_class, axis=1)))
		# print('OG: ' + str(np.argmax(real_l, axis=1)))
		# print('PB: ' + str(np.argmax(fake_l, axis=1)))
		# plt.imshow(np.squeeze(pert[0]), cmap='Greys_r')
		# plt.show(block=False)
		# plt.pause(3)
		# plt.close()


	g_saver.save(sess, "weights/generator/gen.ckpt")
	d_saver.save(sess, "weights/discriminator/disc.ckpt")


def attack(X, y):
	x_pl = tf.placeholder(tf.float32, [None, 28, 28, 1]) # image placeholder

	perturb = generator(x_pl)

	x_perturbed = x_pl + perturb

	d_perturb_logits, d_perturb_probs = discriminator(x_perturbed)

	f = target_model()
	f_real_logits, f_real_probs = f.ModelC(x_pl)
	f_fake_logits, f_fake_probs = f.ModelC(x_perturbed)

	t_vars = tf.trainable_variables()
	f_vars = [var for var in t_vars if 'ModelC' in var.name]
	d_vars = [var for var in t_vars if 'd_' in var.name]
	g_vars = [var for var in t_vars if 'g_' in var.name]

	init  = tf.global_variables_initializer()

	sess = tf.Session()
	sess.run(init)

	f_saver = tf.train.Saver(f_vars)
	g_saver = tf.train.Saver(g_vars)
	d_saver = tf.train.Saver(d_vars)
	f_saver.restore(sess, "./weights/target_model/model.ckpt")
	g_saver.restore(sess, "./weights/generator/gen.ckpt")
	# d_saver.restore(sess, "weights/discriminator/disc.ckpt")

	# p, xp, real_l, fake_l = sess.run([perturb, x_perturbed, f_real_probs, f_fake_probs], \
									# feed_dict={x_pl: X})
	rawpert, pert, fake_l, real_l = sess.run([perturb, x_perturbed, f_fake_probs, f_real_probs], feed_dict={x_pl: X})
	print('LA: ' + str(np.argmax(y, axis=1)))
	print('OG: ' + str(np.argmax(real_l, axis=1)))
	print('PB: ' + str(np.argmax(fake_l, axis=1)))

	print('max: ' + str(np.max(rawpert[0])))
	print('avg: ' + str(np.mean(rawpert[0])))

	plt.imshow(np.squeeze(pert[0]), cmap='Greys_r')
	plt.show()
	# print(p.shape)
	# plt.imshow(p[1,:,:],cmap="Greys_r")





(X,y), (_,_) = mnist.load_data()
X = np.divide(X, 255.0)
X = X.reshape(X.shape[0], 28, 28, 1)
y = to_categorical(y, num_classes=10)

AdvGAN(X, y, batch_size=128)
# rs = np.random.randint(0, X.shape[0], 8)
attack(X[0:8,...], y[0:8,...])





