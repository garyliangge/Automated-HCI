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
    probabilities = []
    labels = []
    # Create the Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="./convnet_model")

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
            predict_tensors = list(classifier.predict(input_fn=eval_input_fn))
            predictions = [c["classes"] for c in predict_tensors]
            probabilities += [c["probabilities"].astype(np.float64).tolist() for c in predict_tensors]
            
            con = tf.confusion_matrix(labels=eval_labels, predictions=predictions, num_classes=7, dtype=tf.int32)
            with tf.Session():
                total_con += np.asarray(tf.Tensor.eval(con,feed_dict=None, session=None))

            accuracies.append(float(eval_results['accuracy']))
            labels += eval_labels.astype(int).tolist()
            print(eval_results)
        
        print(total_con)
        print("OVERALL ACCURACY: {}".format(sum(accuracies) / float(len(accuracies))))

        eval_data = {}
        eval_data["confusion_matrix"] = total_con.tolist()
        eval_data["accuracies"] = accuracies
        eval_data["probabilities"] = probabilities
        eval_data["labels"] = labels

        eval_data_path = './eval_data.json'
        with open(eval_data_path, 'w') as out:
            out.write(json.dumps(eval_data))
 
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
