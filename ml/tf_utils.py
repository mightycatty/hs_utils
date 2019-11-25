import tensorflow as tf
from tensorflow.python.keras import backend as K


def model_wrapper(model_fn, input_shape, model_name):
    input_tensor = tf.keras.layers.Input(input_shape, name='input')
    segmentation_output = model_fn(input_tensor)
    model = tf.keras.models.Model(input_tensor, segmentation_output, name=model_name)
    return model


def convert_keras_model_to_pb(model_fn, weight_path, input_shape, export_path, export_name):
    """
    由于tf keras的底层机制问题，最好不要用参数传递带参数的keras model对象，带参数的模型对象会有一个session，容易导致转pb错误
    注意：该方法导出的pb会多一个import前缀，如keras model下op名字为input, 则pb中为import/input:0
    :param model_with_weights:
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

    K.clear_session()
    K.set_learning_phase(0)  # all new operations will be in test mode from now on
    # serialize the model and get its weights, for quick re-building
    # config = model_with_weights.get_config()
    # weights = model_with_weights.get_weights()
    # re-build a model where the learning phase is now hard-coded to 0
    # new_model = tf.keras.models.Model.from_config(config, custom_objects=custom_obj)
    # new_model.set_weights(weights)
    model = model_wrapper(model_fn, input_shape, export_name)
    model.load_weights(weight_path)
    sess = K.get_session()
    frozen_graph = _freeze_session(sess, output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, export_path, export_name+'.pb', as_text=False)
    return


def calculate_flogs(graph):
    """
    计算flops，传入graph, 当为keras构建时可
        graph = tf.keras.backend.get_graph()
    注意：
        模型构建时不能用none size且需指定batch size，否则无法计算
        img_input = Input(batch_shape=(1, 256, 256, 3))
    :param graph: tensorflow graph
    :return: directly print in terminal
    """
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=graph,
                                run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops


