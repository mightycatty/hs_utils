"""post-training model analyse/optimization
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K


def read_pb(graph_filepath):
    """
    read a pb file, return a graph def obj if success, otherwise returns None
    :param graph_filepath:
    :return: graph_def obj or None
    """
    try:
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def
    except Exception as e:
        print('Pb reading error:{}').format(e)
        return False


def _freeze_session(session, output_node_names, keep_var_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_node_names, freeze_var_names)
        return frozen_graph


def calculate_flogs(graph_or_pb, input_tensor_name=None, input_shape=None):
    """
    cal flops of a tensorflow graph or a frozen.pb
    usage sample:
        # 0. from graph obj
            # explicit batch size is require for a meaningful calculation under this circumstance
            # input_tensor = tklib.placeholder(shape=(1, h, w, c)) explicit batch size of 1
            calculate_flops(graph)
        # 1. from pb file
            calculate_flops(pb, input_tensor_name='old_input_tensor_name:0', (1, 512, 512, 3))
    :param graph_or_pb: tensorflow graph or a frozen pb
    :param input_tensor_name: your original input tensor name
    :param input_shape: input shape with explicit batchsize of 1: (1, h, w, c)
    :return:
    """
    if isinstance(graph_or_pb, str):
        assert input_tensor_name and input_shape, 'input_tensor_name and input_shape is required with a .pb input'
        graph_def = read_pb(graph_or_pb)
        new_input_tensor = tf.placeholder(dtype=tf.float32, shape=input_shape)
        tf.import_graph_def(graph_def, input_map={input_tensor_name: new_input_tensor})
        graph_or_pb = tf.get_default_graph()
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=graph_or_pb,
                                run_meta=run_meta, cmd='scope', options=opts)

    return flops.total_float_ops


def freeze_sess_to_constant_pb(sess, export_name=None, output_node_names=None, as_text=False,
                               dump_result=False, *args, **kwargs):
    """
     output a constant graph for inference and test from a active tklib session
    keep in mind that usually a session in tensorflow if full of duplicate and useless stuff, clean it up before export
    known bug:
        sometime frozen pb only consists of constant node without edges.
    :param sess: activate tklib session with graph and initialized variables
    :param export_name: export name of the .pb file
    :param output_node_names: name of output nodes in graph, auto detect if none given(not 100% safe): node names, not tensor with ':0'
    :param as_text:
    :param dump_result:
    :param args:
    :param kwargs:
    :return: frozen graph_def or False
    """

    def _freeze_session(session, keep_var_names=None, clear_devices=True):
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_node_names, freeze_var_names)
            return frozen_graph

    try:
        if output_node_names is None:
            _, output_node_names = automatic_inputs_outputs_detect(sess.graph.as_graph_def())
        output_node_names = [item.strip(':0') for item in output_node_names]
        frozen_graph = _freeze_session(sess)
        if dump_result:
            tf.io.write_graph(frozen_graph, '.', export_name + '.pb', as_text=as_text)
        return frozen_graph
    except Exception as e:
        print(e)
        return False


def clean_graph_for_inference(graph_or_graph_def, input_node_names, output_node_names):
    """
    trim useless and training-relative nodes for inference.
    do mind that it's merely about graph cleanness, not graph level optimization
    :param graph_or_graph_def:
    :param input_node_names: name of input nodes, str or list
    :param output_node_names:
    :return: graph_def
    """
    from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
    # ================================ graph optimization ==================================
    input_node_names = [input_node_names] if type(input_node_names) is str else input_node_names
    output_node_names = [output_node_names] if type(output_node_names) is str else output_node_names
    input_node_names = [item.strip(':0') for item in input_node_names]
    output_node_names = [item.strip(':0') for item in output_node_names]
    placeholder_type_enum = tf.float32.as_datatype_enum
    if 'GraphDef' not in str(type(graph_or_graph_def)):
        graph_or_graph_def = graph_or_graph_def.as_graph_def()
    graph_def = optimize_for_inference(graph_or_graph_def, input_node_names, output_node_names, placeholder_type_enum)
    return graph_def


def output_pb_to_tensorboard(pb_dir, log_dir):
    """
     RUN:
        tensorboard --logdir=log_dir --host=0.0.0.0
    at the completeness of this function
    :param pb_dir:
    :param log_dir:
    :return: None
    """
    with tf.Session() as sess:
        with tf.gfile.FastGFile(pb_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            graph = tf.import_graph_def(graph_def)
        train_writer = tf.summary.FileWriter(log_dir)
        train_writer.add_graph(sess.graph)


def graph_optimization(frozen_pb_or_graph_def, input_names=None, output_names=None, transforms=None):
    """
    optimize graph for inference
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
    do mind:
        1. output pb is not best for visualization
        2. constants folding is limit in tensorflow graph transforms, with explicit batch size 1 enables more constants folding,
            however still constants not folded.
        3. fix shape for input_tensor is recommanded
    :param frozen_pb_or_graph_def:
    :param input_names: str or list
    :param output_names: str or list
    :param transforms:
    :return: optimize graph def
    """
    try:
        from tensorflow.tools.graph_transforms import TransformGraph
        if transforms is None:
            transforms = [
                'remove_nodes(op=Identity)',
                # 'merge_duplicate_nodes', # not good for visualization
                'strip_unused_nodes',
                # 'remove_attribute(attribute_name=_class)',
                'fold_constants(ignore_errors=true)',
                'fold_batch_norms',
                # 'sort_by_execution_order',
                # 'fuse_convolutions',
                'remove_device',
                # 'quantize_nodes',
                # 'quantize_weights',
            ]
        if isinstance(input_names, str):
            input_names = [input_names]
        if isinstance(output_names, str):
            output_names = [output_names]
        if isinstance(frozen_pb_or_graph_def, str):
            graph_def = read_pb(frozen_pb_or_graph_def)
        else:
            graph_def = frozen_pb_or_graph_def
        if (input_names is None) and (output_names is None):
            input_names, output_names = automatic_inputs_outputs_detect(graph_def)
        optimized_graph_def = TransformGraph(graph_def,
                                             input_names,
                                             output_names,
                                             transforms)
        return optimized_graph_def
    except Exception as e:
        print('graph optimization error:{}'.format(e))
        print('maybe fix the shape of your input_tensor and try again')
        return False


def automatic_inputs_outputs_detect(graph_def):
    """
    automatically detect inputs(nodes with op='Placeholder') and outputs(nodes without output edges) given a graph_def.
    Place note that this is not 100% safe, might yield wrong result, double check before carrying on
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


# TODO: BUG, AVOID TO USE, use freeze_kears_model_to_pb_from_model_fn() instead
def freeze_keras_model_to_pb(tk_model, export_dir='.', export_name=None, optimize_graph=True, verbose=True):
    """
    not safe
    Known issue:
         run into 'SystemError: unknown opcode' if your keras model contains lambda layers
    :param tk_model:
    :param export_dir:
    :param export_name:
    :param optimize_graph:
    :param verbose:
    :return:
    """
    sess_or = K.get_session()
    weights = tk_model.get_weights()
    try:
        with tf.Graph().as_default() as graph, tf.Session(graph=graph).as_default() as sess:
            K.set_session(sess)
            K.set_learning_phase(0)
            new_model = tf.keras.models.clone_model(tk_model)
            new_model.trainable = False
            sess.run(tf.global_variables_initializer())
            new_model.set_weights(weights)
            input_names = [item.name for item in new_model.inputs]
            output_names = [item.name for item in new_model.outputs]
            graph = freeze_sess_to_constant_pb(sess, output_node_names=output_names)
            if export_name:
                export_name = export_name.strip('.pb')
            else:
                export_name = tk_model.name
            tf.io.write_graph(graph, export_dir, export_name + '.pb', as_text=False)
            if optimize_graph:
                graph = graph_optimization(graph, input_names=input_names, output_names=output_names)
                tf.io.write_graph(graph, export_dir, export_name + '.opt.pb')
            if verbose:
                print('frozen pb saved to:{}'.format(os.path.join(export_dir, export_name)))
    except Exception as e:
        print(e)
    finally:
        K.set_session(sess_or)  # rollback keras session
    return


# TODO: code stylish
def freeze_keras_model_to_pb_from_model_fn(model_fn, weight_path, input_shape, export_path, export_name,
                                           graph_optimize=True,
                                           export_tf_lite=False):
    """
    由于tf keras的底层机制问题，最好不要用参数传递带参数的keras model对象，带参数的模型对象会有一个session，容易导致转pb错误
    注意：该方法导出的pb会多一个import前缀，如keras model下op名字为input, 则pb中为import/input:0
    :param model_with_weights:
    :param export_path:
    :param export_name:
    :return:
    """

    def _model_wrapper(model_fn, input_shape, model_name):
        input_tensor = tf.keras.layers.Input(input_shape, name='input')
        segmentation_output = model_fn(input_tensor)
        model = tf.keras.models.Model(input_tensor, segmentation_output, name=model_name)
        return model

    def _freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph

    assert os.path.exists(weight_path), 'weights not found'
    K.clear_session()
    K.set_learning_phase(0)  # all new operations will be in test mode from now on,
    # which is crucial for converting to tflite and a frozen pb
    model = _model_wrapper(model_fn, input_shape, export_name)
    model.load_weights(weight_path, by_name=True)
    with K.get_session() as sess:
        frozen_graph = _freeze_session(sess, output_names=[out.op.name for out in model.outputs])
        tf.io.write_graph(frozen_graph, export_path, export_name + '.pb', as_text=False)
        if graph_optimize:
            frozen_graph = graph_optimization(frozen_graph)
            tf.io.write_graph(frozen_graph, export_path, export_name + '_opt.pb', as_text=False)
        # convert to tflite
        if export_tf_lite:
            converter = tf.lite.TFLiteConverter.from_session(tf.keras.backend.get_session(), model.inputs,
                                                             model.outputs)
            tflite_model = converter.convert()
            save_dir = os.path.join(export_path, export_name + '.tflite')
            with open(save_dir, 'wb') as f:
                f.write(tflite_model)
        return frozen_graph


# TODO
def constant_folding(pb_or_graphdef=None):
    """
    tf graph transform does incomplete constant folding
    :param pb_or_graphdef:
    :return:
    """
    pb_dir = 'F:\heshuai\proj\stylegan\deployment\encoder_fix.pb'
    graph_def = read_pb(pb_dir)
    nodes = graph_def.node
    return graph_def


from ..inference_engine.tfkeras_ie import _load_model


# TODO a more convenient abstract model obj
class ModelAnalyse():
    def __init__(self, model_fn=None, weight_dir=None, input_shape=None, keras_model_with_weights=None, *args,
                 **kwargs):
        def _assert():
            assert model_fn or weight_dir or keras_model_with_weights, 'initialization error'
            if keras_model_with_weights is None:
                assert model_fn and weight_dir and input_shape, 'build_model function/model weights/input_shape ' \
                                                                'are required for initializing a keras model'

        _assert()
        self.model = _load_model(model_fn, weight_dir, input_shape)

    def weight_analyse(self, warning_threshold=1e-7, verbose=False):
        """
        detect weights with extreme value
        :param model:
        :return:
        """
        warning_dict = {}
        for item in self.model.weights:
            weights_value = K.get_value(item)
            w_name = item.name
            abnormal_factor = (np.sum(np.abs(weights_value) > warning_threshold) + \
                               np.sum(np.abs(weights_value) < (1. / warning_threshold)))
            if abnormal_factor > 0:
                warning_dict[w_name] = abnormal_factor / float(weights_value.size)
        warning_dict = sorted(warning_dict.items(), key=warning_dict.values())
        if verbose:
            print(warning_dict)
        return warning_dict

    # TODO
    def convolution_layer_analyse_l1(self):
        pass

    def batch_normalization_layer_analyse(self):
        """
        moving mean and variance in batch normalization layer hits whether outputs from previous are abnormal
        eg, relu+bn
            moving variance < 1e-30 hits relu death zone
        :return:
        """
