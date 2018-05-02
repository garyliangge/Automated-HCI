import os
import json
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
from PIL import Image

category_path = './categories.json'
TEST_SIZE = 1000

"""Returns a label (0, 1, 2) and an image in a numpy ndarray.

For now, images are resized to 400 * 400 and converted to greyscale.
"""
def get_training_batch(size):

	with open(category_path) as json_data:
		categories = json.load(json_data)
		keys = list(categories.keys())[TEST_SIZE:]

		images = np.empty((0, 400*400)).astype(np.float32)
		labels = []

		for _ in range(size):
			index = np.random.randint(0, len(keys))
			screenshot_path = keys[index]
			img = np.asarray(Image.open(screenshot_path).convert('L').resize((400, 400), Image.ANTIALIAS))
			img = img.flatten().astype(np.float32)

			images = np.append(images, [img], axis=0)
			labels.append(categories[screenshot_path])

		return (images, np.asarray(labels, dtype=np.int32))


"""Returns a label (0, 1, 2) and an image in a numpy ndarray."""
def get_eval_data():

	with open(category_path) as json_data:
		categories = json.load(json_data)
		keys = list(categories.keys())[:TEST_SIZE]

		images = np.empty((0, 400*400)).astype(np.float32)
		labels = []

		for screenshot_path in keys:
			img = np.asarray(Image.open(screenshot_path).convert('L').resize((400, 400), Image.ANTIALIAS))
			img = img.flatten().astype(np.float32)

			images = np.append(images, [img], axis=0)
			labels.append(categories[screenshot_path])

		return (images, np.asarray(labels, dtype=np.int32))

