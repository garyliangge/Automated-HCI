import os
import json
import numpy as np
from os import listdir
from os.path import isfile, isdir, join
from PIL import Image
from test import label_counts

data_path = './filtered_traces/'


"""Generate category labels for our screenshots.
We have 10477 samples in totol.
asdfasdf"""
def generate_and_save_labels():

	"""Dictionary that maps from screenshot path to category.
	Categories are:
	0: No action taken
	1: Tap
	2: Swipe
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
							categories[resize_path] = 1
						else:
							xd = abs(d[key][0][0] - d[key][-1][0])
							yd = abs(d[key][0][1] - d[key][-1][1])
							if xd > yd:
								categories[resize_path] = 2
							else:
								categories[resize_path] = 3

	# Save categories in root-level JSON
	category_path = "./categories.json"
	with open(category_path, 'w') as out:
		out.write(json.dumps(categories))
		print("Processed {} samples".format(str(count)))
		print("Labels generated at {}".format(category_path))
		label_counts(categories.values())


if __name__ == '__main__':
	generate_and_save_labels()
