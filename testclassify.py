from getbottleneck import get_bottleneck
import tensorflow as tf
import sys
import os
import numpy as np
import json
import argparse
from tensorflow.python.platform import gfile


parser = argparse.ArgumentParser()
parser.add_argument(
  '--image',
  type=str,
  default='',
  help='Specify where to find the image you wanna test'
)

FLAGS, unparsed = parser.parse_known_args()


image_input = get_bottleneck(FLAGS.image)
image_input = [np.asarray(image_input)]

# Set up saved trained model
with gfile.FastGFile('model/savedgraph.pbtxt','rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
	prediction = sess.graph.get_tensor_by_name('Softmax:0')
	inputs = sess.graph.get_tensor_by_name('inputs:0')
	new_saver = tf.train.import_meta_graph('./model/savedmodel.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint('./model'))
	
	prediction = sess.run(prediction, feed_dict={inputs: image_input})

	print ('Cat score: {}'.format(prediction[0][0]))
	print ('Dog score: {}'.format(prediction[0][1]))