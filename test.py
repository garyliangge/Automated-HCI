from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import json
import numpy as np
import tensorflow as tf
from datamanager import *
from train import cnn_model_fn



# tf.logging.set_verbosity(tf.logging.INFO)

BATCHING = True


def main(unused_argv):
    
    total_con = np.zeros((7,7))

    accuracies = []
    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./convnet_model_simplified")

    if BATCHING:

        # Load training and eval data
        for eval_data, eval_labels in eval_data_batches(10):
            label_counts(eval_labels)

            # Evaluate the model and print results
            eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": eval_data},
                y=eval_labels,
                num_epochs=1,
                shuffle=False)
            eval_results = classifier.evaluate(input_fn=eval_input_fn)
            predictions = [c["classes"] for c in classifier.predict(input_fn=eval_input_fn)]
            con = tf.confusion_matrix(labels=eval_labels, predictions=predictions, num_classes=7, dtype=tf.int32)
            with tf.Session():
                total_con += np.asarray(tf.Tensor.eval(con,feed_dict=None, session=None))
            accuracies.append(eval_results['accuracy'])
            print(eval_results)
        

        print(total_con)
        print("OVERALL ACCURACY: {}".format(sum(accuracies) / float(len(accuracies))))
        confusion_path = './confusion.json'
        with open(confusion_path, 'w') as out:
            out.write(json.dumps(total_con.tolist()))
 
    else:
        # Load training and eval data
        eval_data, eval_labels = get_eval_data()
        label_counts(eval_labels)

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)
        eval_results = classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)



def label_counts(labels):
    counts = [0] * 7
    for label in labels:
        counts[label] += 1
    print("Label counts: {}".format(counts))




if __name__ == "__main__":
    tf.app.run()
