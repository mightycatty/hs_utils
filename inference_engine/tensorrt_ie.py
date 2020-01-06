"""freestanding testing scrip for style-gan encoder and generator with trt backend
requirement: trt 7.0
"""
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning) # disable nasty future warning in tensorflow and numpy
import os
# TODO: run/debug with remote interpreter might encounter .so-not-found error.
#  Adding *TensorRT*/lib to environ path in pycharm fixes this issue
# LD_LIBRARY_PATH=/home/heshuai/TensorRT-7.0.0.11/lib # /home/heshuai/application/TensorRT-7.0.0.11/lib
# print (os.environ)
import tensorrt as trt
import logging
import uff
import numpy as np
import pycuda.driver as cuda

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # global trt logger setting
from PIL.Image import open as img_open
import cv2
import time


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


class TensorrtBuilder:
    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    @staticmethod
    def GiB(val):
        return val * 1 << 30

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

    # deprecated
    @staticmethod
    def build_engine_from_tf_pb_or_uff(model_file, input_node_names, input_node_shapes, output_node_names,
                                       dump_result=True,
                                       uff_text=False, uff_debug=False, max_batch_size=1, max_workspace_size=GiB(1),
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
        builder.fp16_mode = fp16_mode
        built_engine = builder.build_cuda_engine(network)
        if built_engine:
            if dump_result:
                TensorrtBuilder.save_engine(built_engine, name)
            return built_engine
        else:
            logging.error('fail to build engine')
            return False

    @staticmethod
    def build_engine_from_onnx(model_file,
                               dump_result=True,
                               max_batch_size=1,
                               max_workspace_size=GiB(1),
                               debug_sync=True,
                               mixed_precision_mode='float16',
                               explicit_batch_size=False,
                               calib=None,
                               **kwargs):
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
        valid_mixed_precision = ['float32', 'float16', 'int8']
        assert mixed_precision_mode in valid_mixed_precision, 'mixed precision is invalid:{}/{}'.\
            format(mixed_precision_mode, valid_mixed_precision)

        name, model_type = tuple(os.path.splitext(model_file))
        # initialization
        # TODO: BUG, onnx parser fails to parse the model(converted from tf1.14, opset=9) with implicit batch size
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(TensorrtBuilder.EXPLICIT_BATCH) as network, \
                trt.OnnxParser(network, TRT_LOGGER) as parser:
            with open(model_file, 'rb') as model:
                parser.parse(model.read())
                # for i in range(100): # extract error message
                #     print (parser.get_error(i))
            # builder.max_batch_size = max_batch_size
            builder.max_workspace_size = max_workspace_size
            builder.debug_sync = debug_sync
            if mixed_precision_mode == 'int8':
                assert calib is not None, 'calibrator is required for int8 mode'
                builder.int8_mode = True
                builder.int8_calibrator = calib
            if mixed_precision_mode == 'float16':
                builder.fp16_mode = True
            built_engine = builder.build_cuda_engine(network)
            if built_engine:
                if dump_result:
                    TensorrtBuilder.save_engine(built_engine, name)
                return built_engine
            else:
                logging.error('fail to build engine')
                return False


class InferenceWithTensorRT:
    def __init__(self, model_file, pre_processing_fn=None, post_processing_fn=None, force_rebuild=False, **kwargs):
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
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_engine = None
        engine_file = os.path.splitext(self.model_dir)[0] + '.engine'
        if not os.path.exists(engine_file) or self.force_rebuild:
            print ('no built engine found, building a new one...')
            model_type = os.path.splitext(self.model_dir)[-1]
            valid_model_format = ['.pb', '.uff', '.onnx']
            assert model_type in valid_model_format, 'provided model is invalid:{}/{}'.format(model_type, valid_model_format)
            if model_type == '.onnx':
                build_fn = TensorrtBuilder.build_engine_from_onnx
            else:
                build_fn = TensorrtBuilder.build_engine_from_tf_pb_or_uff
            self.trt_engine = build_fn(self.model_dir, **self.kwargs)
        else:
            print ('loading built engine:{}...'.format(engine_file))
            self.trt_engine = TensorrtBuilder.load_engine(self.trt_runtime, engine_file)

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
        if self.pre_processing_fn is not None:
            input_data = self.pre_processing_fn(input_data)
        if str(input_data.dtype) != self.input_dtype.__name__:
            logging.warning('dtype of input data:{} is not compilable with engine input:{}, enforcing dtype convertion'
                            .format(str(input_data.dtype), self.input_dtype.__name__))
            input_data = self.input_dtype(input_data)
        # input data -> cpu
        import time
        time_be = time.time()
        np.copyto(self.host_input, input_data.ravel())
        print (time.time() - time_be)
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


class EncoderInferenceWithTensorRT(InferenceWithTensorRT):
    @staticmethod
    def _load_image(path):
        assert os.path.exists(path)
        image = img_open(path)
        # image = np.transpose(image, [2, 0, 1])
        image_arrays = np.expand_dims(np.array(image), axis=0)
        return image_arrays

    def predict(self, input_data):
        if isinstance(input_data, str):
            input_data = InferenceWithTensorRT._load_image(input_data)
        result = super(EncoderInferenceWithTensorRT, self).predict(input_data)
        result_tf = np.load('./mini_inference/latents_or.npy').flatten()
        return result


class GeneratorInferenceWithTensorRT(InferenceWithTensorRT):
    @staticmethod
    def duplicate_reshape_latents(latents, latent_size=512, n=18):
        """ duplicate the last `latent_size` values of `latents`.
        :param latents: shape[N, latent_size*9]
        :param latent_size: default 512
        :param n: default 18
        :param np_format: return numpy format or not
        :return: shape[N, latent_size*18]
        """
        latents = np.expand_dims(latents, axis=0)
        s = latents.shape
        last_latent_duplicated = np.tile(latents[:, -1 * latent_size:], [1, n - int(s[1] / latent_size)])
        latents_duplicated = np.concatenate((latents, last_latent_duplicated), axis=1)
        return latents_duplicated.reshape((s[0], n, latent_size))

    @staticmethod
    def _saturate_cast(x, drange=[0, 255]):
        x[x > drange[-1]] = drange[-1]
        x[x < drange[0]] = drange[0]
        x = np.uint8(x)
        x = x.reshape((1024, 1024, 3))
        return x

    def predict(self, input_data):
        input_data = GeneratorInferenceWithTensorRT.duplicate_reshape_latents(input_data)
        input_data = np.expand_dims(input_data, axis=0)
        result = super(GeneratorInferenceWithTensorRT, self).predict(input_data)
        result_tf = np.load('./mini_inference/img_or.npy').flatten()
        result = GeneratorInferenceWithTensorRT._saturate_cast(result)
        return result


def inference_performance_eval():
    """
    inference performance evaluation of encoder and generator regardless of pre-post processing
    :return:
    """
    ie_encoder = InferenceWithTensorRT(**build_encoder_kwargs)
    ie_generator = InferenceWithTensorRT(**build_generator_kwargs)
    encoder_input = np.random.randn(1024, 1024, 4).astype(np.float32)
    generator_input = np.random.randn(1, 18, 512).astype(np.float32)
    # warming up
    iterations = 5
    for _ in range(iterations):
        _ = ie_encoder.predict(encoder_input)
        _ = ie_generator.predict(generator_input)

    # performance eval
    print ('performance evaluating...')
    iterations = 1000
    time_be = time.time()
    for _ in range(iterations):
        _ = ie_encoder.predict(encoder_input)
    duration = time.time() - time_be
    print('infer time for encoder:{}'.format(duration / iterations))

    iterations = 1000
    time_be = time.time()
    for _ in range(iterations):
        _ = ie_generator.predict(generator_input)
    duration = time.time() - time_be
    print('infer time for generator:{}'.format(duration / iterations))


def naive_pipeline():
    """
    naive implementation of encoder-generator pipeline
    :return:
    """
    ie_encoder = EncoderInferenceWithTensorRT(**build_encoder_kwargs)
    ie_generator = GeneratorInferenceWithTensorRT(**build_generator_kwargs)
    test_img_dir = 'test.png'
    image = EncoderInferenceWithTensorRT._load_image(test_img_dir)
    dlatent = ie_encoder.predict(image)
    generated_img = ie_generator.predict(dlatent)
    cv2.imwrite('generated_img.png', generated_img)


def generator_test():
    # tensorflow
    from tensorflow_ie import InferenceWithPb
    model_dir = './models/full/generator_fix.opt.pb'
    output_node = 'generator_fix/64x64/Conv1/StyleMod/add_2:0'
    # output_node = 'generator_fix/128x128/Conv0_up/conv2d_transpose:0' #1094.263/-847.3358 vs 1094.2634/-847.3362 (trt)
    # output_node = 'generator_fix/128x128/Conv0_up/Noise/add:0' # 1077.1605/-809.32776 vs 1072.5453/-809.6483
    # output_node = 'generator_fix/128x128/Conv0_up/BiasAdd:0' # 1079.3948/-827.94403 vs 1074.3948/-827.94403
    # output_node = 'generator_fix/images_out:0'
    ie = InferenceWithPb(model_dir)
    input_data = np.ones((1, 1, 18, 512)).astype(np.float32)
    result = ie.predict(input_data, output_nodes=output_node)
    tf_result = np.squeeze(result)

    import matplotlib.pyplot as plt
    import seaborn as sns
    ie = InferenceWithTensorRT(**build_generator_kwargs)
    input_data = np.ones((1, 1, 18, 512)).astype(np.float32)
    result = ie.predict(input_data)
    print (result.shape)
    result = np.squeeze(result)
    print (result.max())
    print (result.min())
    result = np.reshape(result, tf_result.shape)
    diff_or = tf_result - result

    diff = np.mean(diff_or, axis=0)
    f = plt.figure(figsize=(10, 10))
    ax = f.add_subplot(321)
    sns.distplot(diff.flatten(), ax=ax)
    ax = f.add_subplot(322)
    sns.heatmap(diff, xticklabels=False, yticklabels=False, ax=ax)

    diff = np.sum(diff_or, axis=0)
    ax = f.add_subplot(323)
    sns.distplot(diff.flatten(), ax=ax)
    ax = f.add_subplot(324)
    sns.heatmap(diff, xticklabels=False, yticklabels=False, ax=ax)

    diff = np.sum(diff_or / result, axis=0)
    ax = f.add_subplot(325)
    sns.distplot(diff.flatten(), ax=ax)
    ax = f.add_subplot(326)
    sns.heatmap(diff, xticklabels=False, yticklabels=False, ax=ax)

    save_name = '{}.png'.format(output_node.replace('/', '_'))
    plt.savefig(save_name)
    plt.close('all')
    plt.cla()
    plt.clf()
    return result


def encoder_output_check():
    from deployment.tensorflow_ie import InferenceWithPb
    from deployment.onnx_ie import InferenceWithOnnx
    # input_data = np.random.uniform(-1, 1, (1, 4, 1024, 1024)).astype(np.float32)
    input_data = np.ones((1, 4, 1024, 1024)).astype(np.float32)
    # onnx
    model_dir = './encoder_fix.opt.onnx'
    ie_onnx = InferenceWithOnnx(model_dir)
    result = ie_onnx.predict(input_data)
    onnx_result = np.squeeze(result)
    # trt
    ie_trt = InferenceWithTensorRT(**build_encoder_kwargs)
    result = ie_trt.predict(input_data)
    trt_result = np.squeeze(result)
    # tf
    model_dir = './encoder_fix.opt.pb'
    ie_tf = InferenceWithPb(model_dir)
    result = ie_tf.predict(input_data)
    tf_result = np.squeeze(result)
    # liulu
    e = './mini_inference/models/lilu/264_12400_wl_optimized_sgan_encoder.pb'#'./mini_inference/models/lilu/256_optimized_sgan_encoder.pb'
    INPUT_NODE_E = 'E/_Run/images_in:0'
    OUTPUT_NODE_E = 'E/_Run/concat:0'
    ie = InferenceWithPb(e, INPUT_NODE_E, OUTPUT_NODE_E)
    result = ie.predict(input_data)
    ll_result = np.squeeze(result)
    print ('ll:{}'.format(ll_result))
    print('tf:{}'.format(tf_result))
    print ('onnx:{}'.format(onnx_result))
    print ('trt:{}'.format(trt_result))
    return result


if __name__ == '__main__':
    encoder_output_check()

