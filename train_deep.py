from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from datamanager import *



tf.logging.set_verbosity(tf.logging.INFO)
NUM_CLASSES = 7


"""A CNN model based on the tensorflow MNIST tutorial."""

def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	# Our images are 400x400 pixels, and have one color channel (greyscale)
	input_layer = tf.reshape(features["x"], [-1, 400, 400, 1])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.leaky_relu)

	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 400, 400, 32]
	# Output Tensor Shape: [batch_size, 200, 200, 32]
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 200, 200, 32]
	# Output Tensor Shape: [batch_size, 200, 200, 64]
	conv2 = tf.layers.conv2d(
		inputs=pool1,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.leaky_relu)

	# Convolutional Layer #2.5
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 200, 200, 64]
	# Output Tensor Shape: [batch_size, 200, 200, 64]
	conv2_5 = tf.layers.conv2d(
		inputs=conv2,
		filters=64,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.leaky_relu)

	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 200, 200, 64]
	# Output Tensor Shape: [batch_size, 100, 100, 64]
	pool2 = tf.layers.max_pooling2d(inputs=conv2_5, pool_size=[2, 2], strides=2)


	# Convolutional Layer #3
	# Computes 64 features using a 10x10 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 100, 100, 64]
	# Output Tensor Shape: [batch_size, 100, 100, 64]
	conv3 = tf.layers.conv2d(
		inputs=pool2,
		filters=64,
		kernel_size=[10, 10],
		padding="same",
		activation=tf.nn.leaky_relu)

	# Convolutional Layer #3.5
	# Computes 64 features using a 10x10 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 100, 100, 64]
	# Output Tensor Shape: [batch_size, 100, 100, 64]
	conv3_5 = tf.layers.conv2d(
		inputs=conv3,
		filters=64,
		kernel_size=[10, 10],
		padding="same",
		activation=tf.nn.leaky_relu)

	# Pooling Layer #3
	# Second max pooling layer with a 4x4 filter and stride of 4
	# Input Tensor Shape: [batch_size, 100, 100, 64]
	# Output Tensor Shape: [batch_size, 50, 50, 64]
	pool3 = tf.layers.max_pooling2d(inputs=conv3_5, pool_size=[2, 2], strides=2)


	# Convolutional Layer #4
	# Computes 64 features using a 10x10 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 50, 50, 64]
	# Output Tensor Shape: [batch_size, 50, 50, 128]
	conv4 = tf.layers.conv2d(
		inputs=pool3,
		filters=128,
		kernel_size=[5, 5],
		padding="same",
		activation=tf.nn.leaky_relu)

	# Pooling Layer #4
	# Second max pooling layer with a 4x4 filter and stride of 4
	# Input Tensor Shape: [batch_size, 50, 50, 128]
	# Output Tensor Shape: [batch_size, 25, 25, 128]
	pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)


	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 25, 25, 128]
	# Output Tensor Shape: [batch_size, 25 * 25 * 128]
	pool4_flat = tf.reshape(pool4, [-1, 25 * 25 * 128])

	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 25 * 25 * 96]
	# Output Tensor Shape: [batch_size, 1024]
	dense1 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.leaky_relu)

	# Dense Layer
	# Densely connected layer with 512 neurons
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, 512]
	dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.leaky_relu)

	# Dense Layer
	# Densely connected layer with 512 neurons
	# Input Tensor Shape: [batch_size, 512]
	# Output Tensor Shape: [batch_size, 256]
	dense3 = tf.layers.dense(inputs=dense2, units=256, activation=tf.nn.leaky_relu)

	# Add dropout operation; 0.5 probability that element will be kept
	dropout = tf.layers.dropout(
		inputs=dense3, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits layer
	# Input Tensor Shape: [batch_size, 512]
	# Output Tensor Shape: [batch_size, 6]
	logits = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)

	# Avoid NaN loss error by perturbing logits
	epsilon = tf.constant(1e-8)
	logits = logits + epsilon 

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=NUM_CLASSES)
	loss = tf.losses.softmax_cross_entropy(
		onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.04)
		optimizer = tf.train.AdamOptimizer(learning_rate=0.0002)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)




def main(unused_argv):
	# Create the Estimator
	classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, model_dir="./convnet_model_deep")

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=20)

	# Train the model
	for i in range(1000):
	#for train_data, train_labels in eval_data_batches(5):
		print("Loop {}".format(i))
		train_data, train_labels = get_training_batch(300)
		train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": train_data},
			y=train_labels,
			batch_size=30,
			num_epochs=None,
			shuffle=True)
		classifier.train(
			input_fn=train_input_fn,
			steps=50,
			hooks=[logging_hook])



if __name__ == "__main__":
	tf.app.run()
