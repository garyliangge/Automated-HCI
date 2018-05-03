import os
import json
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
from random import shuffle

category_path = './data.json'



"""Returns a label (0, 1, 2) and an image in a numpy ndarray.
For now, images are resized to 400 * 400 and converted to greyscale.
"""
def get_training_batch(size):

	with open(category_path) as json_data:
		train_data, test_data = json.load(json_data)
		keys = list(train_data.keys())

		images = np.empty((0, 400*400)).astype(np.float32)
		labels = np.empty((0, 7)).astype(np.float32)

		for _ in range(size):
			index = np.random.randint(0, len(keys))
			screenshot_path = keys[index]
		#	img = np.asarray(Image.open(screenshot_path).convert('L').resize((400, 400), Image.ANTIALIAS))
			img = np.asarray(Image.open(screenshot_path))
			img = img.flatten().astype(np.float32)

			label = train_data[screenshot_path]
			labels = [0]*7
			labels[train_data[screenshot_path]] = 1.0

			images = np.append(images, [img], axis=0)
			labels = np.append(labels, [label], axis=0)

		return (images, np.asarray(labels, dtype=np.int32))


"""Returns a label (0, 1, 2) and an image in a numpy ndarray."""
def get_eval_data():

	with open(category_path) as json_data:
		train_data, test_data = json.load(json_data)
		keys = list(test_data.keys())

		images = np.empty((0, 400*400)).astype(np.float32)
		labels = np.empty((0, 7)).astype(np.float32)

		for screenshot_path in keys:
		#	img = np.asarray(Image.open(screenshot_path).convert('L').resize((400, 400), Image.ANTIALIAS))
			img = np.asarray(Image.open(screenshot_path))
			img = img.flatten().astype(np.float32)

			label = train_data[screenshot_path]
			labels = [0]*7
			labels[train_data[screenshot_path]] = 1.0

			images = np.append(images, [img], axis=0)
			labels = np.append(labels, [label], axis=0)

		return (images, np.asarray(labels, dtype=np.int32))

"""Returns a generator function for smaller batches."""
def eval_data_batches(num_batches):
	with open(category_path) as json_data:
		train_data, test_data = json.load(json_data)
		data_keys = list(test_data.keys())
		shuffle(data_keys)
		TEST_SIZE = len(test_data)

		for i in range(num_batches):
			keys = data_keys[i*TEST_SIZE//num_batches : (i+1)*TEST_SIZE//num_batches]
			
			images = np.empty((0, 400*400)).astype(np.float32)
			labels = np.empty((0, 7)).astype(np.float32)

			for screenshot_path in keys:
			#	img = np.asarray(Image.open(screenshot_path).convert('L').resize((400, 400), Image.ANTIALIAS))
				img = np.asarray(Image.open(screenshot_path))
				img = img.flatten().astype(np.float32)

				label = train_data[screenshot_path]
				labels = [0]*7
				labels[train_data[screenshot_path]] = 1.0

				images = np.append(images, [img], axis=0)
				labels = np.append(labels, [label], axis=0)

			yield (images, np.asarray(labels, dtype=np.int32))


