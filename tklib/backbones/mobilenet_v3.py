"""standalone MobileNet v3 model scrip, function orient.

most of the code modified from [2](credit to nolanliou)

# Reference
1 .[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
2. https://github.com/nolanliou/mobile-deeplab-v3-plus/blob/master/utils.py
"""

import tensorflow as tf

L2 = tf.keras.regularizers.l2


def _hard_sigmoid(x):
    """
    proposed in Mobilenet-V3(https://arxiv.org/pdf/1905.02244.pdf)

    :param x: input
    :return: output
    """
    return tf.nn.relu6(x + 3) / 6


def _hard_swish(x):
    """
    proposed in Mobilenet-V3(https://arxiv.org/pdf/1905.02244.pdf)

    :param x: input
    :return: output
    """
    return x * _hard_sigmoid(x)


def _global_pool(input_tensor):
    """Applies avg pool to produce 1x1 output.

    NOTE: This function is funcitonally equivalenet to reduce_mean,
    but it has baked in average pool
    which has better support across hardware.

    Args:
      input_tensor: input tensor
    Returns:
      a tensor batch_size x 1 x 1 x depth.
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size = tf.convert_to_tensor(
            [tf.shape(input_tensor)[1],
             tf.shape(input_tensor)[2]])
    else:
        kernel_size = [shape[1], shape[2]]
    output = tf.keras.layers.AvgPool2D(pool_size=kernel_size,
                                       strides=[1, 1],
                                       padding='VALID')(input_tensor)
    # Recover output shape, for unknown shape.
    output.set_shape([None, 1, 1, None])
    return output


def _squeeze_and_excite(x, reduction=4, weight_decay=0.00004):
    scale = _global_pool(x)
    in_channel = x.get_shape().as_list()[-1]
    scale = tf.keras.layers.Dense(
        units=in_channel // reduction,
        activation=tf.nn.relu,
        use_bias=False,
        kernel_regularizer=L2(weight_decay))(scale)
    scale = tf.keras.layers.Dense(
        units=in_channel,
        activation=None,
        use_bias=False,
        kernel_regularizer=L2(weight_decay))(scale)
    scale.set_shape([None, 1, 1, None])
    return x * _hard_sigmoid(scale)


def _depthwiseconv_bn_se(input_tensor,
                   kernel_size=3,
                   depth_multiplier=1,
                   stride=1,
                   padding='SAME',
                   dilation_rate=1,
                   use_bias=False,
                   use_bn=True,
                   bn_momentum=0.997,
                   bn_epsilon=1e-3,
                   use_se=False,
                   activation_fn=tf.nn.relu6,
                   weight_decay=1e-6,
                    ):
    """depthwise conv-bn-squeeze_excite-activation block"""
    net = input_tensor
    net = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride, dilation_rate=dilation_rate,
                                          depth_multiplier=depth_multiplier, use_bias=use_bias, padding=padding,
                                          kernel_regularizer=L2(weight_decay),
                                          bias_regularizer=L2(weight_decay))(net)
    if use_bn:
        net = tf.layers.batch_normalization(
            net,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            )
    if use_se:
        net = _squeeze_and_excite(net)
    if activation_fn:
        net = activation_fn(net)
    return net


def _resize_bilinear(images, size, output_dtype=tf.float32):
    """Returns resized images as output_type.

    Args:
      images: A tensor of size [batch, height_in, width_in, channels].
      size: A 1-D int32 Tensor of 2 elements: new_height, new_width.
            The new size for the images.
      output_dtype: The destination type.
    Returns:
      A tensor of size [batch, height_out, width_out, channels] as a dtype of
        output_dtype.
    """
    images = tf.image.resize_bilinear(images, size, align_corners=True)
    return tf.cast(images, dtype=output_dtype)


def _conv2d(input_tensor,
            num_outputs,
            kernel_size,
            stride=1,
            padding='SAME',
            dilation_rate=1,
            stddev=0.09,
            weight_decay=0.00004,
            use_bias=False,
            use_bn=True,
            bn_momentum=0.997,
            bn_epsilon=1e-3,
            activation_fn=tf.nn.relu,
            ):
    """commonuse conv2d-bn block"""
    net = input_tensor
    conv2d = tf.keras.layers.Conv2D(
        filters=num_outputs,
        kernel_size=kernel_size,
        strides=stride,
        padding=padding,
        dilation_rate=dilation_rate,
        use_bias=use_bias,
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        kernel_regularizer=L2(weight_decay)
        )
    net = conv2d(net)
    if use_bn:
        net = tf.layers.batch_normalization(
            net,
            momentum=bn_momentum,
            epsilon=bn_epsilon)
    if activation_fn:
        net = activation_fn(net)
    return net


def _expanded_conv_mbv3(input_tensor,
                   expansion_size,
                   num_outputs,
                   kernel_size=3,
                   stride=1,
                   padding='SAME',
                   dilation_rate=1,
                   use_se=False,
                   weight_decay=0.00004,
                   activation_fn=tf.nn.relu,
                   ):
    """expanded conv block with squeeze-excite from mobilenet v3"""
    net = input_tensor
    # expansion
    net = _conv2d(net,
                   num_outputs=expansion_size,
                   kernel_size=[1, 1],
                   weight_decay=weight_decay,
                   activation_fn=activation_fn,
                       )
    # depthwise convolution
    net = _depthwiseconv_bn_se(
        net,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation_rate=dilation_rate,
        use_se=use_se,
        activation_fn=activation_fn,
        weight_decay=weight_decay
        )
    # projection
    net = _conv2d(net,
                       num_outputs=num_outputs,
                       kernel_size=[1, 1],
                       weight_decay=weight_decay,
                       activation_fn=None,
                      )
    output_depth = net.get_shape().as_list()[3]
    # shortcut if output has identical shape to input
    input_depth = input_tensor.get_shape().as_list()[3]
    if stride == 1 and input_depth == output_depth:
        net += input_tensor
    return net


def small_mobilenet_v3(x):
    x = _conv2d(x, kernel_size=3, num_outputs=16,
               activation_fn=_hard_swish, stride=2)
    x = _expanded_conv_mbv3(x, kernel_size=3,
                            expansion_size=16, num_outputs=16, use_se=True, stride=2)
    
    x = _expanded_conv_mbv3(x, kernel_size=3,
                            expansion_size=72, num_outputs=24, use_se=False, stride=2)
    x = _expanded_conv_mbv3(x, kernel_size=3,
                            expansion_size=88, num_outputs=24, use_se=False, stride=1)
    
    x = _expanded_conv_mbv3(x, kernel_size=5,
                            expansion_size=96, num_outputs=40, use_se=True, stride=2,
                            activation_fn=_hard_sigmoid)
    x = _expanded_conv_mbv3(x, kernel_size=5,
                            expansion_size=240, num_outputs=40, use_se=True, stride=1,
                            activation_fn=_hard_sigmoid)
    x = _expanded_conv_mbv3(x, kernel_size=5,
                            expansion_size=240, num_outputs=40, use_se=True, stride=1,
                            activation_fn=_hard_sigmoid)
    x = _expanded_conv_mbv3(x, kernel_size=5,
                            expansion_size=120, num_outputs=48, use_se=True, stride=1,
                            activation_fn=_hard_sigmoid)
    x = _expanded_conv_mbv3(x, kernel_size=5,
                            expansion_size=114, num_outputs=48, use_se=True, stride=1,
                            activation_fn=_hard_sigmoid)
    
    x = _expanded_conv_mbv3(x, kernel_size=5,
                            expansion_size=288, num_outputs=96, use_se=True, stride=2,
                            activation_fn=_hard_sigmoid)
    x = _expanded_conv_mbv3(x, kernel_size=5,
                            expansion_size=576, num_outputs=96, use_se=True, stride=1,
                            activation_fn=_hard_sigmoid)
    x = _expanded_conv_mbv3(x, kernel_size=5,
                            expansion_size=576, num_outputs=96, use_se=True, stride=1,
                            activation_fn=_hard_sigmoid)
    return x

