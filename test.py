from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
from datamanager import *
from train import cnn_model_fn



# tf.logging.set_verbosity(tf.logging.INFO)

BATCHING = True


def main(unused_argv):

    accuracies = []

    if BATCHING:
        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./convnet_model")

            # Load training and eval data
        for eval_data, eval_labels in eval_data_batches(10):
            label_counts(eval_labels)

            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data},
                y=eval_labels,
                num_epochs=1,
                shuffle=False)
            eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
            accuracies.append(eval_results['accuracy'])
            print(eval_results)

        print("OVRALL ACCURACY: {}".format(sum(accuracies) / float(len(accuracies))))
        print(eval_results)

    else:
        # Load training and eval data
        eval_data, eval_labels = get_eval_data()
        label_counts(eval_labels)

        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn, model_dir="./convnet_model")

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)



def label_counts(labels):
    counts = [0, 0, 0]
    for label in labels:
        counts[label] += 1
    print("Label counts: {}".format(counts))




if __name__ == "__main__":
    tf.app.run()
