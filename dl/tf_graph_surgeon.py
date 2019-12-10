"""
tensorflow graph processing toolkit
"""
import tensorflow as tf
from tensorflow.python.keras import backend as K


def read_pb(graph_filepath):
    with tf.gfile.GFile(graph_filepath, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def


def calculate_flogs(graph):
    """
    cal flops of a tensorflow graph
    do mind:
        tensor with shape of none should be avoid to make a meaningful calculation.
        eg:
            when construct graph with tf keras api, appoint the full shape(include the batch size axis) to input tensor as below:
                img_input = Input(batch_shape=(1, 256, 256, 3))
    :param graph: tensorflow graph
    :return:
    """
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=graph,
                                run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops


def freeze_sess_to_constant_pb(sess, export_path, export_name, as_text=False):
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
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            input_graph_def = graph.as_graph_def()
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph
    try:
        frozen_graph = _freeze_session(sess)
        tf.train.write_graph(frozen_graph, export_path, export_name+'.pb', as_text=as_text)
        return True
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
    do mind: output pb is not best for visualization
    :param frozen_pb_or_graph_def:
    :param input_names:
    :param output_names:
    :param transforms:
    :return:
    """
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
    from tensorflow.tools.graph_transforms import TransformGraph
    if transforms is None:
        transforms = [
            # 'remove_nodes(op=Identity)',
            'merge_duplicate_nodes',
            'strip_unused_nodes',
            # 'remove_attribute(attribute_name=_class)',
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'sort_by_execution_order',
            'fuse_convolutions',
            'remove_device',
            # 'quantize_nodes',
            # 'quantize_weights',
        ]
    if isinstance(frozen_pb_or_graph_def, str):
        graph_def = read_pb(frozen_pb_or_graph_def)
    else:
        graph_def = frozen_pb_or_graph_def
    optimized_graph_def = TransformGraph(graph_def,
                                         input_names,
                                         output_names,
                                         transforms)
    return optimized_graph_def