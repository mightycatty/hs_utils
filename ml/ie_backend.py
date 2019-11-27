import tensorflow as tf
from tensorflow import gfile
import numpy as np
from tensorflow.python.keras import backend as K


class InferenceWithPb:
    """
    最简单的封装pb模型，供inference用。
    注意没有整合后处理，必须提供已经预处理的数据或者提供预处理函数；输出为tensor直出或者提供后处理函数
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
        # tf1.14下会报内存错误
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

    def predict(self, input_data, **kwargs):
        result = self.sess.run([self.output], feed_dict={self.input: input_data})
        return result

