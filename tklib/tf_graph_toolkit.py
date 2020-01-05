"""
tensorflow-graph toolkit
"""
import tensorflow as tf
from tensorflow.python.tools import freeze_graph

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
                                run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops


def freeze_sess_to_constant_pb(sess, export_name=None, input_node_names=None, output_node_names=None, as_text=False,
                               dump_result=False, *args, **kwargs):
    """
     output a constant graph for inference and test from a active tklib session
    keep in mind that usually a session in tensorflow if full of duplicate and useless stuff, clean it up before export
    known bug:
        sometime frozen pb only consists of constant node without edges.
    :param sess: activate tklib session with graph and initialized variables
    :param export_name: export name of the .pb file
    :param output_node_names: name of output nodes in graph, auto detect if none given(not 100% safe)
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
            # output_names = output_names or []
            # output_names = [v.op.name for v in tklib.global_variables()] # not sure what this does
            input_graph_def = graph.as_graph_def()
            # input_graph_def = clean_graph_for_inference(input_graph_def, input_node_names[:1], output_node_names)
            if clear_devices:
                for node in input_graph_def.node:
                    node.device = ""
            output_names = [item.strip(':0') for item in output_node_names]
            frozen_graph = tf.graph_util.convert_variables_to_constants(
                session, input_graph_def, output_names, freeze_var_names)
            return frozen_graph
    try:
        if output_node_names is None:
            _, output_node_names = automatic_inputs_outputs_detect(sess.graph.as_graph_def())
        if input_node_names is None:
            input_node_names, _ = automatic_inputs_outputs_detect(sess.graph.as_graph_def())
        frozen_graph = _freeze_session(sess)
        if dump_result:
            tf.io.write_graph(frozen_graph, '.', export_name+'.pb', as_text=as_text)
        return frozen_graph
    except Exception as e:
        print (e)
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
    :param frozen_pb_or_graph_def:
    :param input_names: str or list
    :param output_names: str or list
    :param transforms:
    :return: optimize graph def
    """
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


# TODO
def constant_folding(pb_or_graphdef=None):
    pb_dir = 'F:\heshuai\proj\stylegan\deployment\encoder_fix.pb'
    graph_def = read_pb(pb_dir)
    nodes = graph_def.node
    return graph_def

