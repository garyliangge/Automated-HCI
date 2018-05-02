# Automated-HCI
Use CNN's to predict user behavior on mobile UI's

Usage:
1. Download the RICO interaction traces dataset (http://interactionmining.org/rico) and put it in this directory.
2. Run 'python3 generate_labels.py' to generate the screenshot-to-gesture labels in 'categories.json'. The labels are 0 (do nothing), 1 (tap), 2 (swipe).
3. Run 'python3 train.py'.