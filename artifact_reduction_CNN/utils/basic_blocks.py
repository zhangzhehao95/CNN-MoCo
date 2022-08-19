import tensorflow as tf
from tensorflow.keras.layers import add, Layer, InputSpec, Activation, Concatenate, Add, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, BatchNormalization, LayerNormalization, LeakyReLU, PReLU
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization


# -------------------------------------------------------
# reflection padding, taken from https://github.com/fastai/courses/blob/master/deeplearning2/neural-style.ipynb
# -------------------------------------------------------
class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ Default using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super(ReflectionPadding2D, self).get_config()
        config.update({'paddingSize': self.padding})
        return config


class ReflectionPadding3D(Layer):
    def __init__(self, padding=(1, 1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=5)]
        super(ReflectionPadding3D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ Using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3] + 2 * self.padding[1], s[4]

    def call(self, x, mask=None):
        p_pad, w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [p_pad, p_pad], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')

    def get_config(self):
        config = super(ReflectionPadding3D, self).get_config()
        config.update({'paddingSize': self.padding})
        return config


# -------------------------------------------------------
# Conv + BN + ReLU
# -------------------------------------------------------
# 2D convolution layer
def conv2d(input_, filters_=64, kernel_size_=3, strides_=1, pad_='same', name='', mode='BR'):
    """
    mode:
        'F': reflection padding, need to notice the padding mode
        'K': kernel regularizer
        'B': batch normalization
        'I': instance normalization
        'R': ReLU activation
        'L': LeakyReLU
        'P': PReLU activation
    """
    if 'F' in mode:
        pad_size = kernel_size_ // 2
        input_ = ReflectionPadding2D((pad_size, pad_size))(input_)
        pad_ = 'valid'

    if 'B' in mode:
        bias = False
    else:
        bias = True

    if 'K' in mode:
        kr = 1e-7
    else:
        kr = 0

    output_ = Conv2D(filters=filters_, kernel_size=kernel_size_, strides=strides_, padding=pad_, use_bias=bias,
                     kernel_regularizer=tf.keras.regularizers.l2(kr), name='Conv2D/' + name)(input_)

    # Affects the order of operators
    for t in mode:
        if t == 'B':
            output_ = BatchNormalization()(output_)
        elif t == '1':  # LN = GN(groups=1)
            output_ = LayerNormalization()(output_)
        elif t == '2':
            ch_num = output_.shape[-1]
            if ch_num >= 32:
                gn = 32
            else:
                gn = ch_num
            output_ = GroupNormalization(groups=gn, axis=-1)(output_)
        elif t == '3':  # IN = GN(groups=number of channels)
            output_ = InstanceNormalization()(output_)
        elif t == 'R':
            output_ = Activation('relu')(output_)
        elif t == 'L':
            output_ = LeakyReLU(alpha=0.1)(output_)
        elif t == 'P':
            output_ = PReLU()(output_)
        else:
            continue

    return output_


# 3D convolution layer
def conv3d(input_, filters_=64, kernel_size_=3, strides_=1, pad_='same', name='', mode='BR'):
    """
    mode:
        'F': reflection padding, need to notice the padding mode
        'B': batch normalization
        'I': instance normalization
        'R': ReLU activation
    """
    if 'F' in mode:
        pad_size = kernel_size_ // 2
        input_ = ReflectionPadding3D((pad_size, pad_size, pad_size))(input_)
        pad_ = 'valid'

    if 'B' in mode:
        bias = False
    else:
        bias = True

    if 'K' in mode:
        kr = 1e-8
    else:
        kr = 0

    output_ = Conv3D(filters=filters_, kernel_size=kernel_size_, strides=strides_, padding=pad_, use_bias=bias,
                     kernel_regularizer=tf.keras.regularizers.l2(kr), name='Conv3D/' + name)(input_)

    for t in mode:
        if t == 'B':
            output_ = BatchNormalization()(output_)
        elif t == '1':  # LN = GN(groups=1)
            output_ = LayerNormalization()(output_)
        elif t == '2':
            ch_num = output_.shape[-1]
            if ch_num >= 32:
                gn = 32
            else:
                gn = ch_num
            output_ = GroupNormalization(groups=gn, axis=-1)(output_)
        elif t == '3':  # IN = GN(groups=number of channels)
            output_ = InstanceNormalization()(output_)
        elif t == 'R':
            output_ = Activation('relu')(output_)
        elif t == 'L':
            output_ = LeakyReLU(alpha=0.1)(output_)
        elif t == 'P':
            output_ = PReLU()(output_)
        else:
            continue

    return output_


# -------------------------------------------------------
# TransConv + ReLU
# Use transpose conv as up-sampling
# -------------------------------------------------------
# 2D transpose convolution layer
def conv2d_transpose(input_, filters_, kernel_size_=2, strides_=(2, 2), pad_='same', name='', mode=''):
    """
    mode:
        'R': ReLU activation
    """
    output_ = Conv2DTranspose(filters=filters_, kernel_size=kernel_size_, strides=strides_, padding=pad_, name='Conv2DTranspose/' + name)(input_)

    if 'R' in mode:
        output_ = Activation('relu')(output_)
    elif 'L' in mode:
        output_ = LeakyReLU(alpha=0.1)(output_)
    elif 'P' in mode:
        output_ = PReLU()(output_)

    return output_


# 3D transpose convolution layer
def conv3d_transpose(input_, filters_, kernel_size_=2, strides_=(1, 2, 2), pad_='same', name='', mode=''):
    """
    mode:
        'R': ReLU activation
    """
    output_ = Conv3DTranspose(filters=filters_, kernel_size=kernel_size_, strides=strides_, padding=pad_, name='Conv3DTranspose/' + name)(input_)

    if 'R' in mode:
        output_ = Activation('relu')(output_)
    elif 'L' in mode:
        output_ = LeakyReLU(alpha=0.1)(output_)
    elif 'P' in mode:
        output_ = PReLU()(output_)

    return output_


# -------------------------------------------------------
# Residual Dense Block (RDB)
# -------------------------------------------------------
def RDB_block(inp, input_filter=32, inner_filters=16, kernel_size=3, layers=5, name='RDB', mode='R'):
    lists = [inp]
    out = conv2d(input_=inp, filters_=inner_filters, kernel_size_=kernel_size, name=name+'_layer1', mode=mode)
    lists.append(out)
    for i in range(layers - 2):
        concat_in = Concatenate(axis=-1)(lists[:])
        out = conv2d(input_=concat_in, filters_=inner_filters, kernel_size_=kernel_size, name=name + '_layer' + str(i+2), mode=mode)
        lists.append(out)

    # Local feature fusion from the dense net
    concat_in = Concatenate(axis=-1)(lists[:])  # channels_last
    feat = conv2d(input_=concat_in, filters_=input_filter, kernel_size_=1, name=name + '_fusion', mode=mode)
    feat_fusion = Add()([feat, inp])
    return feat_fusion


# -------------------------------------------------------
# Dilated Dense Block (DDB) under 3D case
# -------------------------------------------------------
def DDB3d(inp, channel=16, inner_filter=16, outer_filter=16, kernel_size=3, dense_layers=3, dilate_multi=2, compression=0.5, name='DDB3d'):

    def dilate_bottleneck_layer(in_, inner_filter_, out_filter_, kernel_size_, dilation):
        out = Conv3D(filters=inner_filter_, kernel_size=1, padding='same', use_bias=False)(in_)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        pad_size = (kernel_size_-1) * dilation // 2
        out = ReflectionPadding3D((pad_size, pad_size, pad_size))(out)
        out = Conv3D(filters=out_filter_, kernel_size=kernel_size_, padding='valid', dilation_rate=dilate, use_bias=False)(out)
        out = BatchNormalization()(out)
        out_ = Activation('relu')(out)
        return out_

    def transition_layer(in_, compression_rate):
        ch_num = in_.shape[-1]
        out = Conv3D(filters=int(compression_rate*ch_num), kernel_size=1, padding='same', use_bias=False)(in_)
        out = BatchNormalization()(out)
        out_ = Activation('relu')(out)
        return out_

    dilate = 1
    lists = [inp]
    for i in range(dense_layers):
        if i == 0:
            out = dilate_bottleneck_layer(inp, inner_filter, outer_filter, kernel_size, dilate)
        else:
            concat_in = Concatenate(axis=-1)(lists[:])
            out = dilate_bottleneck_layer(concat_in, inner_filter, outer_filter, kernel_size, dilate)

        lists.append(out)
        dilate *= dilate_multi

    out = Concatenate(axis=-1)(lists[:])
    if compression > 0:
        out = transition_layer(concat_in, compression)
    output = conv3d(input_=out, filters_=channel, kernel_size_=kernel_size, name=name + '_OutputLayer', mode='FBR')
    return output
