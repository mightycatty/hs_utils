import tensorflow as tf
from tensorflow import gfile
import logging
import numpy as np
from tensorflow.python.keras import backend as K


# TODO: supported mode with multi-input-output / infer with batch data
class InferenceWithPb:
    """
    compact tensorflow inference backend which takes in a froze .pb and names of input and output tensor.
        1. data feed to the network is required to be pre-processed beforehand, raw output from network is delivered without any post-processing.
            Or you can register your own preprocessing function and postprocessing function to make a abstract model unit.
        2. tf-trt supported, which tensorflow official claims a 8x speedup(actually not, so try it out for you own good)
        3. every instance of InferenceWithPb create a new session, not using the default session
    """
    def __init__(self, input_name, output_name, pb_dir, pre_processing_fn=None, post_processing_fn=None, tf_trt=False):
        self.input_name = input_name
        self.output_name = output_name
        self.tf_trt = tf_trt
        self.pb_dir = pb_dir
        self.pre_processing_fn = pre_processing_fn
        self.post_processing_fn = post_processing_fn
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
        # tensorrt speed up test
        if self.tf_trt:
            import tensorflow.contrib.tensorrt as trt
            output_name = self.output_name.strip('import/')
            output_name = output_name.strip(':0')
            graph_def = trt.create_inference_graph(
                input_graph_def=graph_def,
                outputs=[output_name],
                precision_mode='FP16',
                max_workspace_size_bytes=1 << 30)
        tf.import_graph_def(graph_def)
        graph = tf.get_default_graph()
        prefix = 'import/'  # tf.import_graph introduces a global name scope: import/
        self.input_name = prefix + self.input_name
        self.output_name = prefix + self.output_name
        self.input = graph.get_tensor_by_name(self.input_name)
        self.output = graph.get_tensor_by_name(self.output_name)

    def _init_session(self):
        # without below configuration, raise error on tf_gpu_1.14
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def predict(self, input_data):
        if self.pre_processing_fn is not None:
            input_data = self.pre_processing_fn(input_data)
        result = self.sess.run([self.output], feed_dict={self.input: input_data})[0]
        if self.post_processing_fn:
            result = self.post_processing_fn(result)
        return result

