import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


model_dir = 'inception'

def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
	bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
	bottleneck_values = np.squeeze(bottleneck_values)
	return bottleneck_values

def maybe_download_and_extract():
	dest_directory = model_dir
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):

		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>> Downloading %s %.1f%%' %(filename, float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()

		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
		print()
		statinfo = os.stat(filepath)
		print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
	tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def create_inception_graph():
	with tf.Session() as sess:
		model_filename = os.path.join(model_dir, 'classify_image_graph_def.pb')
		with gfile.FastGFile(model_filename, 'rb') as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
			bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME, RESIZED_INPUT_TENSOR_NAME]))
	return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def get_bottleneck(image_dir):
	maybe_download_and_extract()
	graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (create_inception_graph())

	sess = tf.Session()

	image_data = gfile.FastGFile(image_dir, 'rb').read()
	return run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
