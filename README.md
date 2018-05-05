# Automated-HCI
Use CNN's to predict user behavior on mobile UI's.

We assume user interactions are broken down into 7 classes:
1. Do nothing
2. Tap top left quadrant
3. Tap top right quadrant
4. Tap bottom left quadrant
5. Tap bottom right quadrant
6. Swipe left/right
7. Swipe up/down

Our training set consists of 10000 UI's for each class. Our testing set consists of 500 UI's for each class.

Usage:
1. Download the RICO interaction traces dataset (http://interactionmining.org/rico) and put it in this directory.
2. Run 'python3 generate_labels.py' to generate the screenshot-to-gesture labels and preprocess images. This can take a while.
3. Run 'python3 train.py', adjusting the step size as the model converges. We recommend starting with a step size of 0.001.
4. Run 'python3 test.py'.
5. To visualize ROC curves, run 'python3 plotting/plot_ROC_curves.py'.
6. To visualize the confusion matrix, run 'python3 plotting/plot_confusion_matrix.py'