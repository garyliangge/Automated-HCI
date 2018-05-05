import os
import json
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
from random import shuffle
from collections import defaultdict
from test import label_counts

data_path = './filtered_traces/'
TEST_SIZE = 3500



"""Generate category labels for our screenshots.
We have 10477 samples in totol.
asdfasdf"""
def generate_and_save_labels():

	"""Dictionary that maps from screenshot path to category.
	Categories are:
	0: No action taken
	1: Tap quadrant 1
	2: Tap quadrant 2
	3: Tap quadrant 3
	4: Tap quadrant 4
	5: Swipe up/down
	6: Swipe left/right
	"""
	categories = {}
	count = 0

	app_paths = [join(data_path, app) for app in listdir(data_path) if isdir(join(data_path, app))]
	for app_path in app_paths:

		#### Used for progress tracking ####
		count += 1
		if not count % 10:
			print("{:.1%}".format(float(count) / len(app_paths)))
		####################################

		trace_paths = [join(app_path, trace) for trace in listdir(app_path) if isdir(join(app_path, trace))]

		for trace_path in trace_paths:
			gesture_path = join(trace_path, "gestures.json")
			screenshot_dir = join(trace_path, "screenshots/")

			with open(gesture_path) as json_data:
				d = json.load(json_data)
				for key in d:
					screenshot_path = join(screenshot_dir, key + ".jpg")


					if isfile(screenshot_path): # Verify the screenshot exists

						resize_path = join(screenshot_dir, key + "_resize.jpg")
						if not isfile(resize_path):
							img = Image.open(screenshot_path).convert('L').resize((400, 400), Image.ANTIALIAS)
							img.save(resize_path)

						if len(d[key]) == 0:
							categories[resize_path] = 0
						elif len(d[key]) == 1:
							x, y = d[key][0]
							if x > 0.5:
								if y > 0.5:
									categories[resize_path] = 1
								else:
									categories[resize_path] = 2
							else:
								if y > 0.5:
									categories[resize_path] = 3
								else:
									categories[resize_path] = 4
						elif len(d[key]) > 1:
							xd = abs(d[key][0][0] - d[key][-1][0])
							yd = abs(d[key][0][1] - d[key][-1][1])
							if xd > yd:
								categories[resize_path] = 5
							else:
								categories[resize_path] = 6

	test = {}
	train = {}
	testcounts = defaultdict(int)
	traincounts = defaultdict(int)
	
	keys = list(categories.keys())
	shuffle(keys)
	for key in keys:
		if testcounts[categories[key]] < TEST_SIZE/7:
			test[key] = categories[key]
			testcounts[categories[key]] += 1
		elif traincounts[categories[key]] < 10000:
			train[key] = categories[key]
			traincounts[categories[key]] += 1

	# Save categories in root-level JSON
	category_path = "./data.json"
	with open(category_path, 'w') as out:
		out.write(json.dumps(((train, test))))
		print("Processed {} samples".format(str(count)))
		print("Labels generated at {}".format(category_path))
		print("\nTrain label counts:")
		label_counts(train.values())
		print("\nTest lable counts:")
		label_counts(test.values())


if __name__ == '__main__':
	generate_and_save_labels()
