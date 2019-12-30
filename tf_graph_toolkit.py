"""
tensorflow-graph involved toolkit and functions
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K
import os


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
        print ('Pb reading error:{}').format(e)
        return False


def calculate_flogs(graph_or_pb, input_tensor_name=None, input_shape=None):
    """
    cal flops of a tensorflow graph or a frozen.pb
    usage sample:
        # 0. from graph obj
            # explicit batch size is require for a meaningful calculation under this circumstance
            # input_tensor = tf.placeholder(shape=(1, h, w, c)) explicit batch size of 1
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
                                run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops


def keras_model_wrapper(model_fn, model_name=None, input_shape=(None, None, 3), trainable=False, batch_size=None, verbose=False):
    """
    wrap model function to a kears model ready for train
    :param model_fn: model fn take input_tensors and delivers corresponding output_tensors
    :param model_name:
    :param input_shape: shape of input tensorflow, default as a RGB image with HWC format
    :param trainable: whether model is for training
    :param batch_size: default none
    :param verbose: display model summary and FLOPs
    :return:
    """
    if verbose:
        batch_size = 1
    tf.keras.backend.set_image_data_format('channels_last')
    input_tensor = tf.keras.layers.Input(input_shape, batch_size=batch_size, name='input')
    segmentation_output = model_fn(input_tensor)
    model = tf.keras.models.Model(input_tensor, segmentation_output, name=model_name, trainable=trainable)
    if verbose:
        graph = K.get_graph()
        calculate_flogs(graph)
        print(model.summary())
        # rebuild model with batch size of none
        # TODO: a more elegant way to do this
        K.clear_session()
        tf.reset_default_graph()
        input_tensor = tf.keras.layers.Input(input_shape, batch_size=batch_size, name='input')
        segmentation_output = model_fn(input_tensor)
        model = tf.keras.models.Model(input_tensor, segmentation_output, name=model_name, trainable=trainable)
    return model


def freeze_sess_to_constant_pb(sess, export_path, export_name, as_text=False, keep_var_names=None, clear_devices=True,
                               output_names=None, *args, **kwargs):
    """
    output a constant graph for inference and test from a active tf session
    keep in mind that usually a session in tensorflow if full of duplicate and useless stuff, clean it up before export
    known bug:
        sometime frozen pb only consists of constant node without edges.
    :param sess:
    :param export_path:
    :param export_name:
    :return:
    """
    def _freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
        graph = session.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
            # output_names = output_names or []
            # output_names = [v.op.name for v in tf.global_variables()] # not sure what this does
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph
    try:
        frozen_graph = _freeze_session(sess, keep_var_names, output_names, clear_devices)
        tf.train.write_graph(frozen_graph, export_path, export_name+'.pb', as_text=as_text)
        return frozen_graph
    except Exception as e:
        print (e)
        return False


def clean_graph_for_inference(graph, input_node_names, output_node_names):
    """
    trim useless and training-relative nodes for inference.
    do mind that it's merely about graph cleanness, not graph level optimization
    :param graph: a constructed graph
    :param input_node_names: name of input nodes, str or list
    :param output_node_names:
    :return: a clean graph
    """
    from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
    # ================================ graph optimization ==================================
    input_node_names = [input_node_names] if type(input_node_names) is str else input_node_names
    output_node_names = [output_node_names] if type(output_node_names) is str else output_node_names
    placeholder_type_enum = tf.float32.as_datatype_enum
    if 'GraphDef' not in str(type(graph)):
        graph = graph.as_graph_def()
    graph_def = optimize_for_inference(graph, input_node_names, output_node_names, placeholder_type_enum)
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


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


def graph_optimization(frozen_pb_or_graph_def, input_names, output_names, transforms=None):
    """
    optimize graph for inference
        # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
    do mind:
        1. output pb is not best for visualization
        2. constants folding is limit in tensorflow graph transforms, with explicit batch size 1 enables more constants folding,
            however still constants not folded.
    :param frozen_pb_or_graph_def:
    :param input_names: str or list
    :param output_names: str or list
    :param transforms:
    :return: optimize graph def
    """
    from tensorflow.tools.graph_transforms import TransformGraph
    if transforms is None:
        transforms = [
            # 'remove_nodes(op=Identity)',
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
    optimized_graph_def = TransformGraph(graph_def,
                                         input_names,
                                         output_names,
                                         transforms)
    return optimized_graph_def


def automatic_inputs_outputs_detect(graph_def):
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
            inputs.append(node.name)
    # outputs detection
    node_inputs = list(set(node_inputs))
    for node in graph_def.node:
        if node.name not in node_inputs:
            if node.input:
                outputs.append(node.name)
    return inputs, outputs


# TODO
def constant_folding(pb_or_graphdef=None):
    pb_dir = 'F:\heshuai\proj\stylegan\deployment\encoder_fix.pb'
    graph_def = read_pb(pb_dir)
    nodes = graph_def.node
    return graph_def


if __name__ == '__main__':
    constant_folding()
