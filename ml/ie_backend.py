import tensorflow as tf
from tensorflow import gfile
import numpy as np
from tensorflow.python.keras import backend as K


class InferenceWithPb:
    """
    compact tensorflow inference backend which takes in a froze .pb and names of input and output tensor
    data feed to the network is required to be pre-processed beforehand, raw output from network is delivered.
    """
    def __init__(self, input_name, output_name, pb_dir):
        self.input_name = input_name
        self.output_name = output_name
        self.pb_dir = pb_dir
        self._construct_graph()
        self._init_session()

    @staticmethod
    def _read_pb(pb_dir):
        with gfile.FastGFile(pb_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    def _construct_graph(self):
        tf.reset_default_graph()
        graph_def = InferenceWithPb._read_pb(self.pb_dir)
        tf.import_graph_def(graph_def)
        graph = tf.get_default_graph()
        self.input = graph.get_tensor_by_name(self.input_name)
        self.output = graph.get_tensor_by_name(self.output_name)

    def _init_session(self):
        # without below configuration, raise error on tf_gpu_1.14
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def predict(self, input_data, **kwargs):
        result = self.sess.run([self.output], feed_dict={self.input: input_data})
        return result

