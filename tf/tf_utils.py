import tensorflow as tf
from tensorflow.python.keras import backend as K


def calculate_flogs(graph):
    """
    cal flops of a graph
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


def freeze_keras_model_to_constant_pb(model_fn, weight_path, input_shape, export_path, export_name):
    """
    given a model_fn and saved_weights of keras model, a constant graph is produced for inference and test
    **do mind** that name of corresponding tensor in keras model will have prefix and suffix,
        eg. "input"  -> "import/input:0"
    :param model_fn:
    :param weight_path:
    :param input_shape:
    :param export_path:
    :param export_name:
    :return: false if any exception and error
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
        K.clear_session()
        K.set_learning_phase(0)  # all new operations will be in test mode from now on
        # serialize the model and get its weights, for quick re-building
        # config = model_with_weights.get_config()
        # weights = model_with_weights.get_weights()
        # re-build a model where the learning phase is now hard-coded to 0
        # new_model = tf.keras.models.Model.from_config(config, custom_objects=custom_obj)
        # new_model.set_weights(weights)
        model = keras_model_wrapper(model_fn=model_fn, input_shape=input_shape, model_name=export_name, verbose=False)
        model.load_weights(weight_path)
        sess = K.get_session()
        frozen_graph = _freeze_session(sess, output_names=[out.op.name for out in model.outputs])
        tf.train.write_graph(frozen_graph, export_path, export_name+'.pb', as_text=False)
        return True
    except Exception as e:
        print (e)
        return False


def freeze_sess_to_constant_pb(sess, export_path, export_name, as_text=False):
    """
    output a constant graph for inference and test from a active tf session
    keep in mind that usually a session in tensorflow if full of duplicate and useless stuff, clean it up before export
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
    :return: a trimmed graph
    """
    from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
    # ================================ graph optimization ==================================
    input_node_names = [input_node_names] if type(input_node_names) is str else input_node_names
    output_node_names = [output_node_names] if type(output_node_names) is str else output_node_names
    placeholder_type_enum = tf.float32.as_datatype_enum
    graph_def = optimize_for_inference(graph.as_graph_def(), input_node_names, output_node_names, placeholder_type_enum)
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph