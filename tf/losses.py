from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf


def expandedSobel(inputTensor):
    sobelFilter = K.variable([[[[1., 1.]], [[0., 2.]], [[-1., 1.]]],
                              [[[2., 0.]], [[0., 0.]], [[-2., 0.]]],
                              [[[1., -1.]], [[0., -2.]], [[-1., -1.]]]])
    inputChannels = K.reshape(K.ones_like(inputTensor[0,0,0,:]),(1,1,-1,1))
    return sobelFilter * inputChannels


def sobel_loss(yTrue,yPred):
    filt = expandedSobel(yTrue)
    yPred = K.expand_dims(yPred[:, :, :, -1], axis=-1)
    sobelTrue = K.depthwise_conv2d(yTrue,filt)
    sobelPred = K.depthwise_conv2d(yPred,filt)
    return K.mean(K.square(sobelTrue - sobelPred))


def ce_sobel_loss(y_true, y_pred, ce_to_edge_factor=1, **kwargs):
    return (ce_to_edge_factor*sparse_categorical_crossentropy(y_true, y_pred)
            + sobel_loss(y_true, y_pred)) / (ce_to_edge_factor + 1.)


# TODO:在python3.5raise错误
def edge_loss(y_true, y_pred, distance='l1'):
    """
    计算mask的边缘差异。边缘采用一阶差分所得，两个方向之和
    :param y_true:
    :param y_pred:
    :return:
    """
    def _first_order(x, axis=1):
        img_nrows = x.shape[1]
        img_ncols = x.shape[2]
        if axis == 1:
            return K.abs(x[:, :img_nrows - 1, :img_ncols - 1] - x[:, 1:, :img_ncols - 1])
        elif axis == 2:
            return K.abs(x[:, :img_nrows - 1, :img_ncols - 1] - x[:, :img_nrows - 1, 1:])
        else:
            return None

    def _calc_loss(pred, target):
        return K.mean(K.square(pred - target))

    y_pred_mask = y_pred[:, :, :, -1]
    # y_pred_mask = K.squeeze(y_pred_mask, axis=-1)
    y_true = K.squeeze(y_true, axis=-1)
    loss = 0
    loss += _calc_loss(_first_order(y_pred_mask, axis=1), _first_order(y_true, axis=1)) + \
             _calc_loss(_first_order(y_pred_mask, axis=2), _first_order(y_true, axis=2))
    return loss


def miou_loss(y_true, y_pred):
    """
    :param y_true:
    :param y_pred: output from softmax
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
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    """
    y_pred = y_pred[:, :, :, -1]
    y_true = K.squeeze(y_true, axis=-1)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


def trimap_loss(y_true, y_pred, recall_factor=0.5, constraint_factor=0.01):
    def _soft_miou_recall(y_true, y_pred):
        # 提取unknown和foreground
        y_pred_un_fore = y_pred[:, :, :, 1:3]
        y_pred = K.sum(y_pred_un_fore, axis=-1)
        y_pred = tf.squeeze(y_pred)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true > 0, tf.int32)
        y_true = tf.cast(y_true, tf.float32)
        y_true = tf.squeeze(y_true)
        intersection = K.sum(K.abs(y_true * y_pred))
        sum_ = K.sum(K.abs(y_true))
        recall = intersection / (sum_ + 1e-5)
        return recall

    def _unknown_constraint(y_pred):
        y_pred_un = y_pred[:, :, :, 1]
        y_pre_fore = y_pred[:, :, :, 2]
        y_pred_un = tf.cast(K.sum(y_pred_un), tf.float32)
        y_pre_fore = tf.cast(K.sum(y_pre_fore), tf.float32)
        constraint = y_pred_un / (y_pred_un + y_pre_fore)
        return constraint
    l_ce = sparse_categorical_crossentropy(y_true, y_pred)
    l_ce = K.mean(l_ce)
    l_recall = _soft_miou_recall(y_true, y_pred)
    l_constraint = _unknown_constraint(y_pred)
    l_sum = tf.cast((1 - recall_factor)*l_ce, tf.float32) + \
            tf.cast(recall_factor*l_recall, tf.float32) + \
            tf.cast(constraint_factor*l_constraint, tf.float32)
    return l_sum


def trimap_alpha_loss_exp25(y_true, y_pred, trimap_factor=1e-2, recall_factor=0.5, constraint_factor=0.01, expand_factor=100.):
    def _alpha_loss():
        y_true_alpha = y_true[:, :, :, 1]
        y_pred_alpha = y_pred[:, :, :, 3]
        l_alpha = tf.keras.losses.mean_squared_error(y_true_alpha, y_pred_alpha)
        return l_alpha
    # trimap loss
    y_true_trimap = y_true[:, :, :, 0]
    y_pred_trimap = y_pred[:, :, :, :3]
    l_trimap = trimap_loss(y_true_trimap, y_pred_trimap, recall_factor=recall_factor, constraint_factor=constraint_factor)
    l_alpha = _alpha_loss()
    # weighted sum
    l_sum = l_trimap*trimap_factor + l_alpha
    return l_sum * expand_factor


def trimap_alpha_loss(y_true, y_pred, trimap_factor=1e-2):
    # trimap loss
    y_true_trimap = y_true[:, :, :, 0]
    y_pred_trimap = y_pred[:, :, :, :3]
    l_ce = sparse_categorical_crossentropy(y_true_trimap, y_pred_trimap)
    l_ce = K.mean(l_ce)
    # alpha loss
    y_true_alpha = y_true[:, :, :, 1]
    y_pred_alpha = y_pred[:, :, :, 3]
    l_alpha = tf.keras.losses.mean_squared_error(y_true_alpha, y_pred_alpha)
    l_sum = l_ce*trimap_factor + l_alpha
    return l_sum


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.executing_eagerly()
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    from skimage import io
    from dataset_utils.img_proccesing import generator_trimap_from_binary_mask
    test_img = r'F:\heshuai\proj\matting_tf\dataset_utils\visualization_samples\annotation\pc_upload_349401_20190424012248000_2.png'
    mask = io.imread(test_img, as_gray=True)
    mask = cv2.resize(mask, (256, 256))
    trimap = generator_trimap_from_binary_mask(mask, band_width=5)
    plt.imshow(trimap)
    plt.show()
    trimap = np.stack([trimap]*3, axis=0)
    trimap = tf.constant(trimap)
    y_pred = np.zeros((3, 256, 256, 3))
    y_pred = tf.constant(y_pred)
    l_sum = trimap_loss(trimap, y_pred)