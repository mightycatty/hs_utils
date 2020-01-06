"""abstract class for inference engine
"""
from abc import ABC, abstractmethod
import onnxruntime as rt
import tensorflow as tf
from tensorflow import gfile
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # required for pycuda initialization
import logging
import os


class InferenceEngine(ABC):
    @abstractmethod
    def __init__(self, model_file, # path to model file
                 input_nodes=None, # input_nodes, str or list or detected automatically if not given
                                    # (some backend do not supported auto input detect)
                 output_nodes=None, # output node, given or auto-detection
                 pre_processing_fn=None, # as named, eg. pre_processing_fn = lambda x: your_fn(x, **kwargs)
                 post_processing_fn=None, #
                 **kwargs):
        pass

    @abstractmethod
    def predict(self,
                input_data, # numpy of list of numpy if multi-inputs
                output_nodes=None, # arbitrary output nodes if backends supported(tf support/onnx not)
                **kwargs):
        pass


class InferenceWithOnnx(InferenceEngine):
    """
    only path to model is required, input and outputs will be detected automatically
    """
    def __init__(self, model_file,
                 pre_processing_fn=None,
                 post_processing_fn=None,
                 **kwargs):
        self.input_names = None
        self.input_shapes = None
        self.output_names = None
        self.output_shapes = None
        self.model_dir = model_file
        self.kwargs = kwargs
        self.pre_processing_fn = pre_processing_fn
        self.post_processing_fn = post_processing_fn
        self._init_session()

    def _get_inputs(self):
        self.input_names = [item.name for item in self.sess.get_inputs()]
        self.input_shapes = [item.shape for item in self.sess.get_inputs()]
        self.output_names = [item.name for item in self.sess.get_outputs()]
        self.output_shapes = [item.shape for item in self.sess.get_outputs()]

    def _init_session(self):
        self.sess = rt.InferenceSession(self.model_dir)
        self._get_inputs()

    def predict(self, input_data,
                output_nodes=None,
                **kwargs):
        if not isinstance(input_data, list):
            input_data = [input_data]
        assert len(input_data) == len(self.input_names), 'num of input_data:{} not matches with what of model\'s ' \
                                                         'input:{}'.format(len(input_data), len(self.input_names))
        feed_dict = {}
        for key, value in zip(self.input_names, input_data):
            feed_dict[key] = value
        result = self.sess.run(self.output_names, input_feed=feed_dict)
        result = result[0] if len(self.output_names) == 1 else result
        return result


class InferenceWithPb(InferenceEngine):
    """
    compact tensorflow inference backend which takes in a froze .pb and names of input and output tensor.
    1. data feed to the network is required to be pre-processed beforehand, raw output from network is delivered without any post-processing.
        Or you can register your own preprocessing function and postprocessing function to make a abstract model unit.
    2. each Inference create a graph and a session for its own, global default graph and session kept untouched.
    3. inputs and outputs is detected automatically if not given, however this is not 100% safe, double check if any unexpectation.
    """

    def __init__(self, model_file,
                 input_nodes=None,
                 output_nodes=None,
                 pre_processing_fn=None,
                 post_processing_fn=None,
                 tf_trt=False,
                 **kwargs):
        # attr
        self.input_name = input_nodes  # list
        self.output_name = output_nodes  # list
        self.input = []  # input tensor
        self.output = []

        self.tf_trt = tf_trt
        self.pb_dir = model_file
        self.pre_processing_fn = pre_processing_fn  # lambda x: fn(x, **kwargs)
        self.post_processing_fn = post_processing_fn
        self._construct_graph()
        self._init_session()


    @staticmethod
    def _read_pb(pb_dir):
        with gfile.FastGFile(pb_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    @staticmethod
    def _automatic_inputs_outputs_detect(graph_def):
        """
        automatically detect inputs(nodes with op='Placeholder') and outputs(nodes without output edges) given a graph_def.
        Place note that this is not 100% safe, might yield wrong inputs outputs detection, double check before carrying on
        :param graph_def:
        :return: inputs(list), outputs(list)
        """
        inputs = []
        outputs = []
        node_inputs = []
        # inputs detection
        for node in graph_def.node:
            node_inputs += node.input
            if node.op == 'Placeholder':
                inputs.append(node.name + ':0')
        # outputs detection
        node_inputs = list(set(node_inputs))
        for node in graph_def.node:
            if node.name not in node_inputs:
                if node.input:
                    outputs.append(node.name + ':0')
        return inputs, outputs

    # @staticmethod
    # def _trt_graph(graph_def, outputs):
    #     graph_def = trt.create_inference_graph(
    #         input_graph_def=graph_def,
    #         outputs=outputs,
    #         precision_mode='FP16',
    #         max_workspace_size_bytes = 1 << 30)
    #     return graph_def

    def _construct_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_def = InferenceWithPb._read_pb(self.pb_dir)
            au_inputs, au_outputs = InferenceWithPb._automatic_inputs_outputs_detect(graph_def)
            tf.import_graph_def(graph_def, name='')
            graph = tf.get_default_graph()
            if self.input_name is None:
                self.input_name = au_inputs
            if self.output_name is None:
                self.output_name = au_outputs
            if isinstance(self.input_name, str):
                self.input_name = [self.input_name]
            if isinstance(self.output_name, str):
                self.output_name = [self.output_name]
            self.input = [graph.get_tensor_by_name(item) for item in self.input_name]
            if self.output_name:
                self.output = [graph.get_tensor_by_name(item) for item in self.output_name]

    def _init_session(self):
        # without below configuration, raise error on tf_gpu_1.14
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.sess = tf.Session(config=config, graph=self.graph)

    def predict(self, input_data,
                output_nodes=None,
                **kwargs):
        output_nodes_list = []
        if output_nodes:
            if isinstance(output_nodes, str):
                output_nodes = [output_nodes]
            assert isinstance(output_nodes, list), 'invalid nodes input:str or list'
            output_nodes_list += output_nodes
        else:
            output_nodes_list += self.output
        if not isinstance(input_data, list):
            input_data = [input_data]
        feed_dict = {key: value for key, value in zip(self.input, input_data)}
        result = self.sess.run(output_nodes_list, feed_dict=feed_dict)
        if len(result) == 1:
            result = result[0]
        return result


class InferenceWithTensorRT:
    def __init__(self, model_file,
                 pre_processing_fn=None,
                 post_processing_fn=None,
                 force_rebuild=False,
                 **kwargs):
        self.model_dir = model_file
        self.pre_processing_fn = pre_processing_fn
        self.post_processing_fn = post_processing_fn
        self.kwargs = kwargs
        self.force_rebuild = force_rebuild
        self._engine_init()
        self._context_init()

    def _engine_init(self):
        """
        load a engine buffer or buid a new one
        :return: a trt engine obj
        """
        logger = trt.Logger(trt.Logger.ERROR)
        self.trt_runtime = trt.Runtime(logger)
        self.trt_engine = None
        engine_file = os.path.splitext(self.model_dir)[0] + '.engine'
        if not os.path.exists(engine_file) or self.force_rebuild:
            print ('no built engine found, building a new one...')
            model_type = os.path.splitext(self.model_dir)[-1]
            valid_model_format = ['.onnx']
            assert model_type in valid_model_format, 'provided model is invalid:{}/{}'.format(model_type, valid_model_format)
            build_fn = InferenceWithTensorRT.build_engine_from_onnx
            self.trt_engine = build_fn(self.model_dir, **self.kwargs)
        else:
            print ('loading built engine:{}...'.format(engine_file))
            self.trt_engine = InferenceWithTensorRT.load_engine(self.trt_runtime, engine_file)

    def _context_init(self):
        volume = trt.volume(self.trt_engine.get_binding_shape(0)) * self.trt_engine.max_batch_size
        self.input_dtype = trt.nptype(self.trt_engine.get_binding_dtype(0))
        self.host_input = cuda.pagelocked_empty(volume, dtype=self.input_dtype)
        volume = trt.volume(self.trt_engine.get_binding_shape(1)) * self.trt_engine.max_batch_size
        dtype = trt.nptype(self.trt_engine.get_binding_dtype(1))
        self.host_output = cuda.pagelocked_empty(volume, dtype=dtype)
        # Allocate device memory for inputs and outputs.
        self.cuda_input = cuda.mem_alloc(self.host_input.nbytes)
        self.cuda_output = cuda.mem_alloc(self.host_output.nbytes)
        self.context = self.trt_engine.create_execution_context()
        self.stream = cuda.Stream()

    @staticmethod
    def save_engine(engine, dump_name) -> bool:
        dump_name = '{}.engine'.format(dump_name) if '.engine' not in dump_name else dump_name
        with open(dump_name, 'wb') as f:
            f.write(engine.serialize())
        return True

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        engine_path = '{}.engine'.format(engine_path) if '.engine' not in engine_path else engine_path
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    @staticmethod
    def build_engine_from_onnx(model_file, dump_result=True, max_batch_size=1, max_workspace_size=1 << 30,
                               debug_sync=True, fp16_mode=True, explicit_batch_size=False, **kwargs):
        """
        trt requirement 7.0
        :param model_file:
        :param dump_result:
        :param max_batch_size:
        :param max_workspace_size:
        :param debug_sync:
        :param fp16_mode:
        :param kwargs:
        :return:
        """
        name, model_type = tuple(os.path.splitext(model_file))
        # initialization
        # TODO: BUG, onnx parser fails to parse the model(converted from tf1.14, opset=9) with implicit batch size
        explicit_batch = 1 << (int)(
            trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # with this setting, network initialized without explicit batch dim
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(explicit_batch) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            with open(model_file, 'rb') as model:
                parser.parse(model.read())
                # for i in range(100):
                #     print (parser.get_error(i))
            # builder.max_batch_size = max_batch_size
            builder.max_workspace_size = max_workspace_size
            builder.debug_sync = debug_sync
            builder.fp16_mode = fp16_mode
            built_engine = builder.build_cuda_engine(network)
            if built_engine:
                if dump_result:
                    InferenceWithTensorRT.save_engine(built_engine, name)
                return built_engine
            else:
                logging.error('fail to build engine')
                return False

    def predict(self, input_data):
        """
        data -> cpu -> GPU -> cpu
        :param input_data:
        :param kwargs:
        :return:
        """
        if self.pre_processing_fn is not None:
            input_data = self.pre_processing_fn(input_data)
        if str(input_data.dtype) != self.input_dtype.__name__:
            logging.warning('dtype of input data:{} is not compilable with engine input:{}, enforcing dtype convertion'
                            .format(str(input_data.dtype), self.input_dtype.__name__))
            input_data = self.input_dtype(input_data)
        # input data -> cpu
        np.copyto(self.host_input, input_data.ravel())
        # cpu -> gpu
        cuda.memcpy_htod_async(self.cuda_input, self.host_input, self.stream)
        # Run inference. difference execution api by the way the engine built(implicit/explicit batch size)
        if self.trt_engine.has_implicit_batch_dimension:
            self.context.execute_async(bindings=[int(self.cuda_input), int(self.cuda_output)], stream_handle=self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=[int(self.cuda_input), int(self.cuda_output)], stream_handle=self.stream.handle)
        # gpu -> cpu.
        cuda.memcpy_dtoh_async(self.host_output, self.cuda_output, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        output = self.host_output
        if self.post_processing_fn is not None:
            output = self.post_processing_fn(output)
        # Return the host output.
        return output