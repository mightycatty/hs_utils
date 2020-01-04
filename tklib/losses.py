"""
custom loss for tensorflow kears
"""
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf


def ce_sobel_loss(y_true, y_pred, ce_to_edge_factor=1, **kwargs):
    def _sobel_loss(yTrue, yPred):
        def _expandedSobel(inputTensor):
            sobelFilter = K.variable([[[[1., 1.]], [[0., 2.]], [[-1., 1.]]],
                                      [[[2., 0.]], [[0., 0.]], [[-2., 0.]]],
                                      [[[1., -1.]], [[0., -2.]], [[-1., -1.]]]])
            inputChannels = K.reshape(K.ones_like(inputTensor[0, 0, 0, :]), (1, 1, -1, 1))
            return sobelFilter * inputChannels

        filt = _expandedSobel(yTrue)
        yPred = K.expand_dims(yPred[:, :, :, -1], axis=-1)
        sobelTrue = K.depthwise_conv2d(yTrue, filt)
        sobelPred = K.depthwise_conv2d(yPred, filt)
        return K.mean(K.square(sobelTrue - sobelPred))
    return (ce_to_edge_factor*sparse_categorical_crossentropy(y_true, y_pred)
            + _sobel_loss(y_true, y_pred)) / (ce_to_edge_factor + 1.)


def miou_loss(y_true, y_pred):
    """
    miou loss for binary segmentation
    :param y_true: [B, H, W, 1]
    :param y_pred: [B, H, W, C], C=2 stands for binary segmentaiton
    :return:
    """
    y_pred = y_pred[:, :, :, -1]
    # y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.cast(y_true, tf.float32)
    y_true = K.squeeze(y_true, axis=-1)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    miou = (intersection) / (sum_ - intersection)
    return 1 - miou


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    loss for segmentation dealing with unbalanced class samples
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    """
    y_pred = y_pred[:, :, :, -1]
    y_true = K.squeeze(y_true, axis=-1)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coef_loss(y_true, y_pred):
    """
    loss for segmentation dealing with unbalanced class samples
    :param y_true:
    :param y_pred:
    :return:
    """
    def _dice_coef(y_true, y_pred, smooth=1):
        """
        Dice = (2*|X & Y|)/ (|X|+ |Y|)
             =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
        ref: https://arxiv.org/pdf/1606.04797v1.pdf
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
    return 1-_dice_coef(y_true, y_pred)