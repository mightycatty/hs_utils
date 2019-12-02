import numpy as np
from tensorflow.python.keras import backend as K


def keras_weight_analyse(model, warning_threshold=1e-6, log_file='weights_abnormal.log', clear_log=True):
    """
    raise warning if abnormal weights found(abs of which exceeds given warning_threshold)
    :param model: activate keras model with weights
    :return:
    """
    from logger import MyLog
    weights_logger = MyLog(log_file)
    if clear_log:
        weights_logger.clear_logfile()
    warning_dict = {}
    abnormal_layers = []
    for item in model.weights:
        weights_value = K.get_value(item)
        w_name = item.name
        warning_flag = np.sum(np.abs(weights_value) < warning_threshold) > 0
        if warning_flag:
            warning_dict[w_name] = weights_value
            weights_logger.info(w_name)
            layer_item = w_name.split('/')[0]
            abnormal_layers.append(layer_item)
    return list(set(abnormal_layers))