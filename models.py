import numpy as np
import tensorflow as tf

def conv2d(x, hidden_size, kernel_size, stride=1, pooling_size=0,
		pooling_stride=2, dropout=0.0, activation=None, training=False):
	"""
	Args:
        x: Tensor with shape [batch, dim_x, dim_y]
        hidden_size: Int. The number of filters in this layer.
        kernel_size: Int. The width of the kernels in this layer.
        stride: Int. How much to stride the filters.
        pooling_size: Int. How much to pool the outputs of the conv layer.
        dropout: Float. Probability to keep an activation.
        activation: An activation function. If None, linear activation.
	"""
	net = tf.layers.conv2d(
	        inputs=x,
	        filters=hidden_size,
	        kernel_size=kernel_size,
	        padding="same",
	        activation=activation)
	if pooling_size:
		net = tf.layers.max_pooling2d(net, [pooling_size, pooling_size],
                  padding="same", strides=pooling_stride)
	return tf.layers.dropout(net, dropout, training=training)

def fc(x, n_units, dropout, activation=None, training=False):
    """Fully connected layer with dropout.

    Args:
        x: Tensor with shape [batch, dimensions].
        n_units: Int. Number of output units.
        dropout: Float. Probability to keep an activation.
        activation: An activation function. If None, linear activation.

    Returns:
        The output activation of the layer.
    """
    net = tf.layers.dense(x, n_units)
    net = tf.contrib.layers.layer_norm(net)
    if activation:
        net = activation(net)
    return tf.layers.dropout(net, dropout, training=training)

def cnn_hp(**kwargs):
    """Constructs a default set of hyperparameters for a CNN.

    Args:
        kwargs: keyword arguments to override defaults.

    Returns:
        An HParam object to construct a CNN.
    """
    hp = tf.contrib.training.HParams()
    hp.n_conv_layers = 4
    hp.hidden_sizes = [128,128,128,64]
    hp.fc_h_size = 925
    hp.kernel_size = 8
    hp.pooling_sizes = [2, 2, 2, 4]
    hp.stride = 1
    hp.drop_probs = [0., 0., 0., 0.]
    hp.dropout = 0.
    hp.activation = lrelu
    hp.output_activation = tf.sigmoid
    hp.__dict__.update(kwargs)
    return hp

def cnn(input_, n_classes, hp, training=False):
	net = input_
	for i in range(hp.n_conv_layers):
        net = conv2d(net, hp.hidden_sizes[i], hp.kernel_size, hp.stride,
                     pooling_size=hp.pooling_sizes[i], pooling_stride=hp.pooling_stride,
                     dropout=hp.drop_probs[i], activation=hp.activation,
                     training=training)
    net = tf.contrib.layers.flatten(net)
    net = fc(net, hp.fc_h_size, hp.dropout, activation=hp.activation, 
        training=training)
    return fc(net, n_classes, hp.dropout, activation=hp.output_activation,
        training=training)

def save_hparams(hparams_file, hparams):
  """Save hparams."""
  with codecs.getwriter("utf-8")(tf.gfile.GFile(hparams_file, "wb")) as f:
    f.write(hparams.to_json())

def load_hparams(hparams_file):
  """Load hparams from an existing model directory."""
  if tf.gfile.Exists(hparams_file):
    with codecs.getreader("utf-8")(tf.gfile.GFile(hparams_file, "rb")) as f:
      try:
        hparams_values = json.load(f)
        hparams = tf.contrib.training.HParams(**hparams_values)
      except ValueError:
        print_out("  can't load hparams file")
        return None
    return hparams
  else:
    return None


def build_graph(features, labels, num_logits, hp=cnn_hp(), mode):
	input_placeholder = tf.reshape(features["x"], [-1, 400, 400, 1])
	logits = cnn(input_placeholder, num_logits, hp, training)
	training = tf.placeholder(dtype = tf.bool)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    return {"input_placeholder": input_placeholder,
     		"target_placeholder": labels,
     		"optimizer": optimizer,
    		"logits": logits,
    		"loss": loss,
    		"init_op": init_op,
            "training": training,
     		"saver": saver}

def build_estimator_graph(features, labels, mode, hp=cnn_hp()):
	input_placeholder = tf.reshape(features["x"], [-1, 400, 400, 1])
	logits = cnn(input_placeholder, num_logits, hp, training)
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
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
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