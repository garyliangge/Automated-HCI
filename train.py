from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import glob
import logz
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import tensorflow as tf
from datamanager import *
from models.py import build_estimator_graph



tf.logging.set_verbosity(tf.logging.INFO)
NUM_CLASSES = 7

def train(ops, iterations, train_iterator, valid_iterator,
          log_freq, save_freq,):
    with tf.Session() as sess:
        # If a model exists, restore it. Otherwise, initialize a new one
        if glob.glob(save_path + '*'):
            ops["saver"].restore(sess, save_path)
            print("Weights restored.")
        else:
            sess.run(ops["init_op"])
            print("Weights initialized.")

        # The training iterator
        
        training_losses = []
        valid_losses = []

        for i in range(iterations):
            input_, target = next(train_iterator)
            # One step of training
            _loss, _ = sess.run([ops["loss"], ops["optimizer"]], feed_dict={
                ops["input_placeholder"]: input_,
                ops["training"]: True
            })

            training_losses.append(_loss)

            # Save the model
            if i % save_freq == 0:
                save_path = ops["saver"].save(sess, save_path)
                print("Model saved in file: %s" % save_path)

            # Validate and log results
            if i % log_freq == 0:
                all_logits = np.array([[0] * num_logits])
                all_targets = np.array([[0] * num_logits])

                # The validation loop
                for _ in range(valid_size):
                    b, d, t = next(train_iterator)
                    _logits, _valid_loss = sess.run([ops["logits"], ops["loss"]],
                     feed_dict={
                        ops["input_placeholder"]: b,
                        ops["dnase_placeholder"]: d,
                        ops["target_placeholder"]: t,
                        ops["training"]: False})
                    valid_losses += [_valid_loss]
                    all_logits = np.append(all_logits, _logits, axis = 0)
                    all_targets = np.append(all_targets, t, axis = 0)

                # Log relevant statistics
                log(i, training_losses, valid_losses, all_logits, all_targets,
                    num_logits)
                training_losses = []
                valid_losses = []

def log(i,
        training_losses,
        valid_losses,
        valid_logits,
        valid_targets,
        num_logits
        ):
    """Logging a single gradient step to outfile.

    Args:
        i: Int. Current gradient step iteration.
        training_losses: Float. The training loss.
        validation_losses: Float. The validation loss.
        valid_logits: To comput ROC AUC.
        valid_targets: To comput ROC AUC.

    Returns:
        Nothing. Logs get dumped to outfile.
    """
    aucs = []
    for j in np.arange(num_logits):
        try:
            aucs += [roc_auc_score(valid_targets[:, j],valid_logits[:, j])]
        except ValueError:
            continue

    logz.log_tabular('Iteration', i)
    logz.log_tabular('Loss', np.mean(training_losses))
    logz.log_tabular('Valid Loss', np.mean(valid_losses))
    logz.log_tabular('Average AUPRC', np.mean(aucs))
    logz.log_tabular('80th percentile AUPRC', np.percentile(aucs, 80))
    logz.dump_tabular()

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
		activation=tf.nn.relu)

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
		activation=tf.nn.relu)

	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 200, 200, 64]
	# Output Tensor Shape: [batch_size, 100, 100, 64]
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)


	# Convolutional Layer #3
	# Computes 64 features using a 10x10 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 100, 100, 64]
	# Output Tensor Shape: [batch_size, 100, 100, 64]
	conv3 = tf.layers.conv2d(
		inputs=pool2,
		filters=32,
		kernel_size=[10, 10],
		padding="same",
		activation=tf.nn.relu)

	# Pooling Layer #3
	# Second max pooling layer with a 4x4 filter and stride of 4
	# Input Tensor Shape: [batch_size, 100, 100, 32]
	# Output Tensor Shape: [batch_size, 25, 25, 32]
	pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[4, 4], strides=4)


	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 25, 25, 32]
	# Output Tensor Shape: [batch_size, 25 * 25 * 32]
	pool3_flat = tf.reshape(pool3, [-1, 25 * 25 * 32])

	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 25 * 25 * 96]
	# Output Tensor Shape: [batch_size, 1024]
	dense1 = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.leaky_relu)

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
		optimizer = tf.train.AdamOptimizer(learning_rate=0.00002)
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



<<<<<<< HEAD
=======
def main(unused_argv):
    # Load training and eval data
    train_data, train_labels = get_training_batch(200)
    eval_data, eval_labels = get_eval_data()

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=build_estimator_graph, model_dir="./convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=2)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=20,
        num_epochs=None,
        shuffle=True)
    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
>>>>>>> temp

def main(unused_argv):
	# Create the Estimator
	classifier = tf.estimator.Estimator(
		model_fn=cnn_model_fn, model_dir="./convnet_model")

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
			steps=25,
			hooks=[logging_hook])

if __name__ == "__main__":
	tf.app.run()
