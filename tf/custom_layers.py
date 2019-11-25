"""
custom layers and block built with tf.keras api
** create a custom layer with Lamda to avoid the need to provide custom objects when loading model(seems that even with
lamda to create a custom layer still yield error when loading model)
"""
import tensorflow as tf
from tensorflow.python.keras.layers import Conv2D, BatchNormalization, Add, Activation, DepthwiseConv2D, Lambda, ReLU, \
    ZeroPadding2D
from tensorflow.python.keras import backend as K, backend, layers
import cv2
from dataset_utils.img_processing_tf import tf_get_edge


def dense_block_sepconv(x, nb_layers, nb_filter, depth_activation=False, relu6=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    concat_feat = x
    for i in range(nb_layers):
        x = SepConv_BN(concat_feat, nb_filter, prefix='dense_decoder_'+str(i), depth_activation=depth_activation,
                       relu6=relu6)
        concat_feat = tf.keras.layers.Concatenate(axis=-1)([x, concat_feat])
    return concat_feat


def dense_block_sepconv_sigmoid(x, nb_layers, nb_filter):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    def _SepConv_BN_sigmoid(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=True, epsilon=1e-3,
                   relu6=False):
        """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
            Implements right "same" padding for even kernel sizes
            Args:
                x: input tensor
                filters: num of filters in pointwise convolution
                prefix: prefix before name
                stride: stride at depthwise conv
                kernel_size: kernel size for depthwise convolution
                rate: atrous rate for depthwise convolution
                depth_activation: flag to use activation between depthwise & poinwise convs
                epsilon: epsilon to use in BN layer
        """
        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'

        if not depth_activation:
            x = tf.keras.layers.Activation('sigmoid')(x)
        x = tf.keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride),
                                            dilation_rate=(rate, rate),
                                            padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = tf.keras.layers.Activation('sigmoid')(x)
        x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same',
                                   use_bias=False, name=prefix + '_pointwise')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = tf.keras.layers.Activation('sigmoid')(x)
        return x
    concat_feat = x
    for i in range(nb_layers):
        x = _SepConv_BN_sigmoid(concat_feat, nb_filter, prefix='dense_decoder_'+str(i), depth_activation=True)
        concat_feat = tf.keras.layers.Concatenate(axis=-1)([x, concat_feat])
    return concat_feat


def dense_block_sepconv_swish(x, nb_layers, nb_filter):
    def _SepConv_BN_swish(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=True, epsilon=1e-3):
        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'

        if not depth_activation:
            x = tf.keras.layers.Activation(h_swish)(x)
        x = tf.keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride),
                                            dilation_rate=(rate, rate),
                                            padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = tf.keras.layers.Activation(h_swish)(x)
        x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same',
                                   use_bias=False, name=prefix + '_pointwise')(x)
        x = tf.keras.layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = tf.keras.layers.Activation(h_swish)(x)
        return x
    concat_feat = x
    for i in range(nb_layers):
        x = _SepConv_BN_swish(concat_feat, nb_filter, prefix='dense_decoder_'+str(i), depth_activation=True)
        concat_feat = tf.keras.layers.Concatenate(axis=-1)([x, concat_feat])
    return concat_feat


def MBConv_idskip(x_input, filters, kernel_size, strides=1, filters_multiplier=1, alpha=1., rate=1):
    """ Mobile inverted bottleneck convolution (Block b, c, d, e of MNasNet paper https://arxiv.org/pdf/1807.11626.pdf)

    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        alpha: An integer which multiplies the filters dimensionality
    # Returns
        Output tensor.
    """

    depthwise_conv_filters = make_divisible(x_input.shape[3].value)
    pointwise_conv_filters = make_divisible(filters * alpha)

    x = conv_bn(x_input, filters=depthwise_conv_filters * filters_multiplier, kernel_size=1, strides=1)
    x = depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides, rate=rate)
    x = conv_bn(x, filters=pointwise_conv_filters, kernel_size=1, strides=1, activation=False)

    # Residual connection if possible
    if strides == 1 and x.shape[3] == x_input.shape[3]:
        return tf.keras.layers.add([x_input, x])
    else:
        return x


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_bn(x, filters, kernel_size, strides=1, alpha=1, activation=True):
    """
    Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    """
    filters = make_divisible(filters * alpha)
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                      use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.9)(x)
    if activation:
        x = tf.keras.layers.ReLU(max_value=6)(x)
    return x


def depthwiseConv_bn(x, depth_multiplier, kernel_size, strides=1, rate=1):
    """ Depthwise convolution
    The  tf.keras.layers.DepthwiseConv2D is just the first step of the Depthwise Separable convolution (without the pointwise step).
    Depthwise Separable convolutions consists in performing just the first step in a depthwise spatial convolution
    (which acts on each input channel separately).

    This function defines a 2D Depthwise separable convolution operation with BN and relu6.
    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, depth_multiplier=depth_multiplier,
                               padding='same', use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(l=0.0003),
                                        dilation_rate=(rate, rate))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.9)(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)
    return x


def sepConv_bn_noskip(x, filters, kernel_size, strides=1):
    """ Separable convolution block (Block F of MNasNet paper https://arxiv.org/pdf/1807.11626.pdf)

    # Arguments
        x: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """
    x = depthwiseConv_bn(x, depth_multiplier=1, kernel_size=kernel_size, strides=strides)
    x = conv_bn(x, filters=filters, kernel_size=1, strides=1)
    return x


def SepConv_BN(x, filters, prefix, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3, relu6=False,
               weight_decay=0.):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """
    max_relu_value = 6 if relu6 else None
    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = tf.keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = tf.keras.layers.ReLU(max_value=max_relu_value)(x)
    x = tf.keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                        padding=depth_padding, use_bias=False, name=prefix + '_depthwise', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, (1, 1), padding='same',
               use_bias=False, name=prefix + '_pointwise', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
    if depth_activation:
        x = tf.keras.layers.ReLU(max_value=max_relu_value)(x)
    return x


def inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1, sequeeze_factor=1):
    in_channels = K.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)
    if block_id:
        # Expand
        if sequeeze_factor > 1:
            x = Conv2D(int(in_channels / sequeeze_factor), kernel_size=1, padding='same',
                       use_bias=True, activation=None,
                       name=prefix + 'squeeze')(x)
        x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
                   use_bias=False, activation=None,
                   name=prefix + 'expand')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.9,
                               name=prefix + 'expand_BN')(x)
        x = ReLU(max_value=6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'
    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        name=prefix + 'depthwise')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.9,
                           name=prefix + 'depthwise_BN')(x)

    x = ReLU(max_value=6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               name=prefix + 'project')(x)
    x = BatchNormalization(epsilon=1e-3, momentum=0.9,
                           name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])

    # if in_channels == pointwise_filters and stride == 1:
    #    return Add(name='res_connect_' + str(block_id))([inputs, x])

    return x


def large_kernel_conv(input_tensor, k=11, filter_num=None):
    """
    旷世的文章，large kernel matters, 利用两个k/1和1/K的卷积逼近K*K的，可以增大感受野
    :param input_tensor:
    :param k:
    :return:
    """
    if filter_num is None:
        filter_num = K.int_shape(input_tensor)[-1]
    b_0 = tf.keras.layers.Conv2D(filter_num, (1, k), padding='same',  activation=None)(input_tensor)
    b_0 = tf.keras.layers.Conv2D(filter_num, (k, 1), padding='same',  activation=None)(b_0)
    b_1 = tf.keras.layers.Conv2D(filter_num, (k, 1), padding='same', activation=None)(input_tensor)
    b_1 = tf.keras.layers.Conv2D(filter_num, (1, k), padding='same', activation=None)(b_1)
    b_sum = tf.keras.layers.Add()([b_0, b_1])
    b_sum = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.9)(b_sum)
    b_sum = tf.keras.layers.ReLU(max_value=6)(b_sum)
    return b_sum


def dense_block_sepconv_large_kernel(x, nb_layers, nb_filter, depth_activation=False, relu6=False):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    concat_feat = x
    for i in range(nb_layers):
        x = large_kernel_conv(concat_feat)
        x = SepConv_BN(x, nb_filter, prefix='dense_decoder_'+str(i), depth_activation=depth_activation,
                       relu6=relu6)
        concat_feat = tf.keras.layers.Concatenate(axis=-1)([x, concat_feat])
    return concat_feat


def resize_with_tensor_pair(inputs):
    """
    输入两个tensor，将前后resize成后者的大小, 返回resize后的前者
    :param inputs:
    :return:
    """
    x, x_desired = inputs
    size_before = K.shape(x_desired)
    x = tf.image.resize_images(x, size_before[1:3],
                           method=tf.image.ResizeMethod.BILINEAR,
                           align_corners=True)
    return x


def spatial_attention_fusion(sp_feature, cnt_feature):
    """
    以语义层次的特征生成辅助mask和edge，对低层次特征进行attention,然后通过add进行高低层次语义信息融合
    :param sp_feature:
    :param cnt_feature:
    :return:
    """
    # attention to enhance spatial feature
    sp_feature_channel = K.int_shape(sp_feature)[-1]
    cnt_feature_channel = K.int_shape(cnt_feature)[-1]
    sp_feature = tf.keras.layers.Conv2D(filters=cnt_feature_channel, kernel_size=1, strides=1, padding='same',
                      use_bias=False)(sp_feature)
    # attention multiplier/ mask - edge
    mask = tf.keras.layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', use_bias=False)(cnt_feature)
    mask = Lambda(resize_with_tensor_pair)([mask, sp_feature])
    mask_multiplier = tf.keras.layers.Concatenate(axis=-1)([mask] * cnt_feature_channel)
    edge = tf.keras.layers.Lambda(lambda xx: tf_get_edge(xx, 10))(mask)
    edge_multiplier = tf.keras.layers.Concatenate(axis=-1)([edge] * cnt_feature_channel)
    # cnt feature processing
    cnt_feature = Lambda(resize_with_tensor_pair)([cnt_feature, sp_feature])
    # mask attention
    mask_attention = tf.keras.layers.Multiply()([mask_multiplier, sp_feature])
    mask_attention = tf.keras.layers.Add()([cnt_feature, mask_attention])
    # edge attention
    edge_attention = tf.keras.layers.Multiply()([edge_multiplier, sp_feature])
    edge_attention = tf.keras.layers.Add()([cnt_feature, edge_attention])
    # fusion
    feature_fusion = tf.keras.layers.Concatenate(axis=-1)([edge_attention, mask_attention])
    feature_fusion = tf.keras.layers.Conv2D(32, (1, 1), padding='same', activation='sigmoid', use_bias=False)(feature_fusion)
    return feature_fusion


def channel_attention(input_tensor):
    """
    参考[1]的channel attention block，用于语义特征的attention
    疑问：
    bn是否有必要
    https://arxiv.org/pdf/1808.00897.pdf
    :param input_tensor:
    :return:
    """
    x = input_tensor
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = Lambda(lambda x: K.expand_dims(x, 1))(x)
    x = Lambda(lambda x: K.expand_dims(x, 1))(x)
    input_channel_num = K.int_shape(x)[-1]
    x = tf.keras.layers.Conv2D(input_channel_num, (1, 1), use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-3, momentum=0.9)(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, input_tensor])
    return x


def tf_gabor_conv(input_tensor, ksize=(3, 3), stride=(1, 1), sigma=1.0, theta=0, lambd=15.0, gamma=0.02):
    # Create a 3x3 Gabor filter
    params = {'ksize': ksize, 'sigma': sigma, 'theta': theta, 'lambd': lambd , 'gamma': gamma}
    filter = cv2.getGaborKernel(**params)
    # make the filter to have 4 dimensions.
    filter = tf.expand_dims(filter, 0)
    filter = tf.expand_dims(filter, 3)
    # Apply the filter on `image`
    output_result = K.conv2d(input_tensor, filter, strides=stride, padding='SAME')
    return output_result


def conv2d_same(x, filters, prefix, stride=1, kernel_size=3, rate=1):
    """Implements right 'same' padding for even kernel sizes
        Without this there is a 1 pixel drift when stride = 2
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
    """
    if stride == 1:
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='same', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        return Conv2D(filters,
                      (kernel_size, kernel_size),
                      strides=(stride, stride),
                      padding='valid', use_bias=False,
                      dilation_rate=(rate, rate),
                      name=prefix)(x)


def xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                   rate=1, depth_activation=False, return_skip=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            return_skip: flag to return additional tensor after 2 SepConvs for decoder
            """
    residual = inputs
    for i in range(3):
        residual = SepConv_BN(residual,
                              depth_list[i],
                              prefix + '_separable_conv{}'.format(i + 1),
                              stride=stride if i == 2 else 1,
                              rate=rate,
                              depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                               kernel_size=1,
                               stride=stride)
        shortcut = BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    if return_skip:
        return outputs, skip
    else:
        return outputs


def resize_to_same_shape(inputs):
    """
    输入两个tensor,将前者通过双线性插值到后者的大小，返回resize后的前者
    :param inputs:
    :return:
    """
    x, x_before = inputs
    size_before = K.shape(x_before)
    x = tf.image.resize_images(x, size_before[1:3],
                           method=tf.image.ResizeMethod.BILINEAR,
                           align_corners=True)
    return x


def resize_to_given_shape(inputs):
    """
    输入两个tensor,将前者通过双线性插值到后者的大小，返回resize后的前者
    :param inputs:
    :return:
    """
    x, width, height = inputs
    x = tf.image.resize_images(x, (height, width),
                           method=tf.image.ResizeMethod.BILINEAR,
                           align_corners=True)
    return x

def h_swish(x):
    """
    mobilenet v3中的h-swish activation
    :param x:
    :return:
    """
    x = x*K.relu(x+3, max_value=6) / 6.
    return x


def model_wrapper(model_fn, model_name, shape=(None, None, 3), trainable=False, batch_size=None):
    tf.keras.backend.set_image_data_format('channels_last')
    input_tensor = tf.keras.layers.Input(shape, batch_size=batch_size, name='input')
    segmentation_output = model_fn(input_tensor)
    model = tf.keras.models.Model(input_tensor, segmentation_output, name=model_name, trainable=trainable)
    return model


if __name__ == '__main__':
    import numpy as np
    tf.enable_eager_execution()
    tf.executing_eagerly()
    input_tensor = tf.constant(np.ones((3, 32, 32, 256),dtype=np.float32))
    output = channel_attention(input_tensor)


