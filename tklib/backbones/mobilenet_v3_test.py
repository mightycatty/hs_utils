import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, ReLU, Lambda, Activation, Concatenate, Dropout, \
    Softmax, GlobalAveragePooling2D, DepthwiseConv2D, Add

l2 = L2 = tf.keras.regularizers.l2


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, name=None):
    return tf.nn.relu6(x + 3, name) / 6


def hard_swish(x):
    x = x * hard_sigmoid(x)
    return x


def kept_dim_global_pool(x):
    """Applies avg pool to produce 1x1 output.
    Args:
      input_tensor: input tensor
    Returns:
      a tensor batch_size x 1 x 1 x depth.
    """
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    x = Lambda(lambda x_:K.expand_dims(x_, axis=1))(x)
    x = Lambda(lambda x_:K.expand_dims(x_, axis=1))(x)
    return x


def squeeze_and_excite(x, reduction=4, weight_decay=0.00004):
    scale = kept_dim_global_pool(x)
    in_channel = x.get_shape().as_list()[-1]
    # scale = tf.keras.layers.Dense(
    #     units=in_channel // reduction,
    #     activation=tf.nn.relu,
    #     use_bias=False,
    #     kernel_regularizer=L2(weight_decay))(scale)
    scale = tf.keras.layers.Conv2D(filters=in_channel // reduction, kernel_size=1, use_bias=False,
                                   activation=tf.nn.relu, kernel_regularizer=l2(weight_decay))(scale)
    # scale = tf.keras.layers.Dense(
    #     units=in_channel,
    #     activation=hard_sigmoid,
    #     use_bias=False,
    #     kernel_regularizer=L2(weight_decay))(scale)
    scale = tf.keras.layers.Conv2D(filters=in_channel, kernel_size=1, use_bias=False,
                                  activation=hard_sigmoid, kernel_regularizer=l2(weight_decay))(scale)
    # scale = Lambda(hard_sigmoid)(scale)
    x = tf.keras.layers.Multiply()([x, scale])
    return x


def inverted_res_block(inputs, stride, alpha, filters, block_id, skip_connection, rate=1, sequeeze_factor=1,
                       use_se=False, weight_decay=1e-6, activation_fn=tf.nn.relu6, depthconv_kernel_size=3,
                       expansion=6, expansion_filters=None):
    """
    basic block for building mobilenet

    details:
      1. projection conv2d activation=None
    :param inputs:
    :param expansion:
    :param stride:
    :param alpha:
    :param filters:
    :param block_id:
    :param skip_connection:
    :param rate:
    :param sequeeze_factor:
    :return:
    """
    in_channels = K.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = make_divisible(pointwise_conv_filters, 8)
    if expansion_filters is None:
        expansion_filters = make_divisible(int(expansion * in_channels), 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand
        if sequeeze_factor > 1:
            x = Conv2D(int(in_channels / sequeeze_factor), kernel_size=1, padding='same',
                       use_bias=True, activation=None,
                       name=prefix + 'squeeze',
                       kernel_regularizer=l2(weight_decay))(x)
        x = Conv2D(expansion_filters, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.9,
                               name=prefix + 'expand_BN')(x)
        x = Lambda(activation_fn, name=prefix + 'expand_activation')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=depthconv_kernel_size, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise',
                        kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.9,
                           name=prefix + 'depthwise_BN')(x)

    x = Lambda(activation_fn, name=prefix + 'depthwise_activation')(x)
    # squeeze and excite
    if use_se:
        x = squeeze_and_excite(x, weight_decay=weight_decay)
    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.9,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        output_depth = x.get_shape().as_list()[3]
        # add shortcut if output has identical shape to input
        input_depth = inputs.get_shape().as_list()[3]
        if stride == 1 and input_depth == output_depth:
            return Add(name=prefix + 'add')([inputs, x])
    return x


def mobilev3_deeplabv3p_model_fn_test(input_tensor, alpha=1., weight_decay=1e-8):
    """
    """
    x = input_tensor
    x = Conv2D(16,
               kernel_size=3,
               strides=(2, 2), padding='same',
               use_bias=False, name='Conv')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.9, name='Conv_BN')(x)
    x = Lambda(hard_swish, name='Conv_hard_swish')(x)

    x = inverted_res_block(x, filters=16, alpha=alpha, stride=2,
                           expansion_filters=16, block_id='head_0', skip_connection=False, use_se=True,
                           weight_decay=weight_decay)
    #
    x = inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                           expansion_filters=72, block_id=1, skip_connection=False, use_se=False,
                           weight_decay=weight_decay)

    x = inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                           expansion_filters=88, block_id=2, skip_connection=True, use_se=False,
                           weight_decay=weight_decay)

    kernel_size = 5
    x = inverted_res_block(x, filters=40, alpha=alpha, stride=2,
                           expansion_filters=96, block_id=3, skip_connection=False, use_se=True,
                           activation_fn=hard_sigmoid, depthconv_kernel_size=kernel_size, weight_decay=weight_decay)
    x = inverted_res_block(x, filters=40, alpha=alpha, stride=1,
                           expansion_filters=240, block_id=4, skip_connection=True, use_se=True,
                           activation_fn=hard_sigmoid, depthconv_kernel_size=kernel_size, weight_decay=weight_decay)
    x = inverted_res_block(x, filters=40, alpha=alpha, stride=1,
                           expansion_filters=240, block_id=5, skip_connection=True, use_se=True,
                           activation_fn=hard_sigmoid, depthconv_kernel_size=kernel_size, weight_decay=weight_decay)
    x = inverted_res_block(x, filters=48, alpha=alpha, stride=1,
                           expansion_filters=120, block_id=6, skip_connection=True, use_se=True,
                           activation_fn=hard_sigmoid, depthconv_kernel_size=kernel_size, weight_decay=weight_decay)
    x = inverted_res_block(x, filters=48, alpha=alpha, stride=1,
                           expansion_filters=114, block_id=7, skip_connection=True, use_se=True,
                           activation_fn=hard_sigmoid, depthconv_kernel_size=kernel_size, weight_decay=weight_decay)

    kernel_size = 5
    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=1,
                           expansion_filters=288, block_id=8, skip_connection=False, use_se=True,
                           activation_fn=hard_sigmoid, depthconv_kernel_size=kernel_size, weight_decay=weight_decay)
    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=1,
                           expansion_filters=576, block_id=9, skip_connection=True, use_se=True,
                           activation_fn=hard_sigmoid, depthconv_kernel_size=kernel_size, weight_decay=weight_decay)

    x = inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=1,
                           expansion_filters=576, block_id=10, skip_connection=True, use_se=True,
                           activation_fn=hard_sigmoid, depthconv_kernel_size=kernel_size, weight_decay=weight_decay)

    x = Conv2D(128, (1, 1), padding='same',
               use_bias=False, name='backbone_feature', activation=None, kernel_regularizer=l2(weight_decay))(x)
    return x