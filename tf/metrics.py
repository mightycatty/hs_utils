import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import control_flow_ops
import numpy as np


def mean_iou(y_true, y_pred):
    """
    当预测输出只有一个通道，计算广义miou，否则认为是softmax输出，取argmax后计算binary
    :param y_true: gt with shape of [b, h, w, 1]
    :param y_pred: [b, h, w, 1/N]
    :return:
    """
    y_pred = tf.cast(y_pred, tf.float32)
    channel = tf.shape(y_pred)[-1]
    y_pred = control_flow_ops.cond(
        pred=tf.math.equal(channel, 1),
        true_fn=lambda: y_pred,
        false_fn=lambda: tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    )
    y_pred = tf.squeeze(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.squeeze(y_true)
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    miou_fg = intersection / (sum_ - intersection + 1e-5)
    # background
    y_pred = tf.ones_like(y_pred) - y_pred
    y_true = tf.ones_like(y_true) - y_true
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    miou_bg = intersection / (sum_ - intersection + 1e-5)
    miou = (miou_bg + miou_fg) / 2.
    return miou


def mean_iou_trimap_alpha(y_true, y_pred):
    """
    计算前后景的平均miou
    :param y_true:
    :param y_pred: softmax out
    :return:
    """
    y_pred = y_pred[:, :, :, -1]
    y_pred = K.expand_dims(y_pred, axis=-1)
    y_true = y_true[:, :, :, -1]
    miou = mean_iou(y_true, y_pred)
    return miou


def foreground_miou_binary(y_true, y_pred):
    """
    forground miou for binary segmentation
    :param y_true: mask of [height, width, 1]/[height, width]
    :param y_pred: sigmoid output of mask of [height, width, 1/2]
    :return:
    """
    channel = tf.shape(y_pred)[-1]
    y_pred = control_flow_ops.cond(
        pred=tf.math.equal(channel, 1),
        true_fn=lambda: tf.to_int32(y_pred > 0.5),
        false_fn=lambda: tf.to_int32(tf.argmax(y_pred, axis=-1))
    )
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.squeeze(y_true)
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    miou = intersection / (sum_ - intersection + 1e-5)
    return miou


def foreground_miou_binary_trimap_alpha(y_true, y_pred):
    """
    forground miou for binary segmentation
    :param y_true: mask of [height, width, 1]/[height, width]
    :param y_pred: sigmoid output of mask of [height, width, 1/2]
    :return:
    """
    y_pred = y_pred[:, :, :, -1]
    y_pred = K.expand_dims(y_pred, axis=-1)
    y_true = y_true[:, :, :, -1]
    channel = tf.shape(y_pred)[-1]
    y_pred = control_flow_ops.cond(
        pred=tf.math.equal(channel, 1),
        true_fn=lambda: tf.to_int32(y_pred > 0.5),
        false_fn=lambda: tf.to_int32(tf.argmax(y_pred, axis=-1))
    )
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.squeeze(y_true)
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    miou = intersection / (sum_ - intersection + 1e-5)
    return miou


def merge_miou_trimap(y_true, y_pred):
    """
    合并unknown foreground作为前景,gt前景miou
    :param y_true: mask of [height, width, 1]/[height, width]
    :param y_pred: sigmoid output of mask of [height, width, 3]
    :return:
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred > 0, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.cast(y_true > 0, tf.int32)
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.squeeze(y_true)
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    miou = intersection / (sum_ - intersection + 1e-5)
    return miou


def foreground_miou_trimap(y_true, y_pred):
    """
    foreground作为前景,gt前景miou
    :param y_true: mask of [height, width, 1]/[height, width]
    :param y_pred: sigmoid output of mask of [height, width, 1/2]
    :return:
    """
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred > 1, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.cast(y_true > 1, tf.int32)
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.squeeze(y_true)
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    miou = intersection / (sum_ - intersection + 1e-5)
    return miou


def merge_miou_recall(y_true, y_pred):
    """
    trimap输出下,合并unknown foreground作为前景,gt前景的recall miou
    :param y_true: mask of [height, width, 3]/[height, width]
    :param y_pred: sigmoid output of mask of [height, width, 3]
    :return:
    """
    y_pred = tf.to_int32(tf.argmax(y_pred, axis=-1))
    y_pred = tf.cast(y_pred > 0, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.cast(y_true > 0, tf.int32)
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.squeeze(y_true)
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true))
    recall = intersection / (sum_ + 1e-5)
    return recall


def merge_miou_recall_trimap_alpha(y_true, y_pred):
    """
    trimap输出下,合并unknown foreground作为前景,gt前景的recall miou
    :param y_true: mask of [height, width, 3]/[height, width]
    :param y_pred: sigmoid output of mask of [height, width, 3]
    :return:
    """
    y_true = y_true[:, :, :, -1]
    y_pred = y_pred[:, :, :, :3]
    y_pred = tf.to_int32(tf.argmax(y_pred, axis=-1))
    y_pred = tf.cast(y_pred > 0, tf.int32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.squeeze(y_pred)
    y_true = tf.cast(y_true > 0, tf.int32)
    y_true = tf.cast(y_true, tf.float32)
    y_true = tf.squeeze(y_true)
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true))
    recall = intersection / (sum_ + 1e-5)
    return recall