"""
almost-standalone tensorrt inference scrip for converting and inferring tensorflow frozen model (pb).
"""
import os
# TODO: run/debug with remote interpreter might encounter .so-not-found error.
#  Adding *TensorRT*/bin to environ path in pycharm fixes this issue
# print (os.environ)
import tensorrt as trt
import logging
import uff
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # TODO: this is required for fixing no-active-context error


class EasyDict(dict):
    from typing import Any
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


# global configuration

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
build_kwargs = EasyDict()
# build_kwargs.model_file = 'encoder_frozen.uff'
build_kwargs.model_file = 'encoder_frozen.pb'
# build_kwargs.input_node_names = 'import/import/encoder_input:0'
build_kwargs.input_node_names = 'encoder_input'
build_kwargs.input_node_shapes = (1, 4, 1024, 1024)
# build_kwargs.output_node_names = 'import/import/latents_out:0'
build_kwargs.output_node_names = 'latents_out'
build_kwargs.dump_result = True
build_kwargs.uff_text = True
# build config
build_kwargs.max_batch_size = 1
build_kwargs.max_workspace_size = 1 << 20
build_kwargs.debug_sync = True
build_kwargs.fp16_mode = True


def _exception_logger_wrapper(func):
    from functools import wraps
    import logging
    @wraps(func)
    def newfunc(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(e)
            return False
    return newfunc


@_exception_logger_wrapper
def save_engine(engine, dump_name) -> bool:
    dump_name = '{}.engine'.format(dump_name) if '.engine' not in dump_name else dump_name
    with open(dump_name, 'wb') as f:
        f.write(engine.serialize())
    return True


@_exception_logger_wrapper
def load_engine(trt_runtime, engine_path):
    engine_path = '{}.engine'.format(engine_path) if '.engine' not in engine_path else engine_path
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def build_engine_from_tf_pb_or_uff(model_file, input_node_names, input_node_shapes, output_node_names,
                                   dump_result=False,
                                   uff_text=False, uff_debug=False, max_batch_size=1, max_workspace_size=1 << 20,
                                   debug_sync=True, fp16_mode=True):
    """
     build TRT engine from a frozen tensorflow .pb model or a scrip-converted uff model.
    reminder: prefixes and suffixes are not required
        eg： input_node_names = 'import/encoder_input:0'         X
            input_node_names = 'encoder_input'                   √
    :param model_file:
    :param input_node_names:
    :param input_node_shapes:
    :param output_node_names:
    :param dump_result:
    :param uff_text:
    :param uff_debug:
    :param max_batch_size:
    :param max_workspace_size:
    :param debug_sync:
    :param min_find_iterations:
    :param average_find_iterations:
    :param fp16_mode:
    :return:
    """
    name, model_type = tuple(os.path.splitext(model_file))
    supported_type_list = ['.pb', '.uff']
    assert model_type in supported_type_list, 'support model list:{}'.format(supported_type_list)
    # wrap input and output kwargs to list for universal multi-nodes engine building
    if isinstance(input_node_names, str):
        input_node_names = [input_node_names]
        input_node_shapes = [input_node_shapes]
    output_node_names = [output_node_names] if isinstance(output_node_names, str) else output_node_names
    output_node_names = [output_node_names] if isinstance(output_node_names, str) else output_node_names
    # convert pb model to uff
    if model_type == '.pb':
        uff_buffer = uff.from_tensorflow_frozen_model(frozen_file=model_file, output_nodes=output_node_names,
                                                      output_filename=name + '.uff', text=uff_text,
                                                      debug_mode=uff_debug)
    # initialization
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.UffParser()
    # parse network
    for input_node_name, input_node_shape in zip(input_node_names, input_node_shapes):
        parser.register_input(input_node_name, input_node_shape)
    for output_node_name in output_node_names:
        parser.register_output(output_node_name)
    if model_type == '.pb':
        parser.parse_buffer(uff_buffer, network)
    else:
        parser.parse(model_type, network)
    # build engine
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = max_workspace_size
    builder.debug_sync = debug_sync
    # builder.min_find_iterations = min_find_iterations
    # builder.max_find_iterations = max_find_iterations
    # builder.average_find_iterations = average_find_iterations
    builder.fp16_mode = fp16_mode
    built_engine = builder.build_cuda_engine(network)
    if dump_result:
        save_engine(built_engine, name)
        return built_engine


# TODO: multi inputs and outputs support
class InferenceWithTensorRT:
    def __init__(self, model_file, pre_processing_fn=None, post_processing_fn=None, **kwargs):
        self.model_dir = model_file
        self.pre_processing_fn = pre_processing_fn
        self.post_processing_fn = post_processing_fn
        self.kwargs = kwargs
        self._engine_init()
        self._context_init()

    def _engine_init(self):
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_engine = None
        engine_file = os.path.splitext(self.model_dir)[0] + '.engine'
        if not os.path.exists(engine_file):
            self.trt_engine = build_engine_from_tf_pb_or_uff(**self.kwargs)
        else:
            self.trt_engine = load_engine(self.trt_runtime, engine_file)

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

    def predict(self, input_data):
        """
        data -> cpu -> GPU -> cpu
        :param input_data:
        :param kwargs:
        :return:
        """
        if str(input_data.dtype) != self.input_dtype.__name__:
            logging.warning('dtype of input data:{} is not compilable with engine input:{}, enforcing dtype convertion'
                            .format(str(input_data.dtype), self.input_dtype.__name__))
            input_data = self.input_dtype(input_data)
        # input data -> cpu
        np.copyto(self.host_input, input_data.ravel())
        # cpu -> gpu
        cuda.memcpy_htod_async(self.cuda_input, self.host_input, self.stream)
        # Run inference.
        self.context.execute_async(bindings=[int(self.cuda_input), int(self.cuda_output)], stream_handle=self.stream.handle)
        # gpu -> cpu.
        cuda.memcpy_dtoh_async(self.host_output, self.cuda_output, self.stream)
        # Synchronize the stream
        self.stream.synchronize()
        # Return the host output.
        return self.host_output


def test():
    ie = InferenceWithTensorRT(**build_kwargs)
    image = np.ones((1, 4, 1024, 1024), dtype=np.float32)
    import time
    time_be = time.time()
    iteration = 1000
    for i in range(iteration):
        result = ie.predict(image)
    duration = time.time() - time_be
    print ('mean infer time:{}'.format(duration / iteration))


if __name__ == '__main__':
    build_engine_from_tf_pb_or_uff(**build_kwargs)
    # test()
