from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Subtract, MaxPool2D, MaxPool3D, Dropout, UpSampling2D, UpSampling3D
from tensorflow.keras.activations import linear
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from utils.optimizer import Optimizer
from utils.metrics import structural_similarity, peak_snr
from utils.basic_blocks import *


def mixed_loss(L1_weight, L2_weight):
    def mixed(y_true, y_pred):
        return L1_weight * mean_absolute_error(y_true, y_pred) + L2_weight * mean_squared_error(y_true, y_pred)
    return mixed


class UNet_3D:
    def __init__(self, cf):
        self.model = None
        self.cf = cf
        self.kernel_size = cf.kernel_size  # kernal size along each dimension, 3
        self.filters_base = cf.filter_num_base  # original filter number, 21
        self.up_down_times = cf.up_down_times  # number of levels - 1, 3
        self.conv_times = cf.conv_times  # number of convs each level, 2

    def make(self):
        # poolSize[0] = Stride[0] = 1 if no down-sampling on slice dimension
        poolSize = (2, 2, 2)
        stride = (2, 2, 2)

        conv_mode = 'F'
        if self.cf.weight_decay:
            conv_mode += 'K'

        if self.cf.norm == 'Batch':
            conv_mode += 'B'
        elif self.cf.norm == 'Layer':
            conv_mode += '1'
        elif self.cf.norm == 'Group':
            conv_mode += '2'
        elif self.cf.norm == 'Instance':
            conv_mode += '3'
        else:
            pass

        if self.cf.activation == 'ReLU':
            conv_mode += 'R'
        elif self.cf.activation == 'LeakyReLU':
            conv_mode += 'L'
        elif self.cf.activation == 'PReLU':
            conv_mode += 'P'
        else:
            pass

        skip_connection = []
        # Input layer
        model_input = Input(shape=self.cf.model_shape, name='UNet3D/Input')
        x = model_input

        # Down sampling
        for layer in range(self.up_down_times):
            filters = (2 ** layer) * self.filters_base
            # The filter number for each convolution belonging to the same level stays the same
            for i in range(self.conv_times):
                x = conv3d(x, filters, self.kernel_size, name='Down_%d/Layer_%d' % (layer, i), mode=conv_mode)
            skip_connection.append(x)
            # Down-sampling
            if self.cf.downsample == 'StridedConv':
                x = conv3d(x, filters, self.kernel_size, stride, name='Down_%d/StridedConv' % layer, mode='')
            else:
                x = MaxPool3D(pool_size=poolSize, strides=stride, name='Down_%d/MaxPool3D' % layer)(x)

        # Bottom layer
        filters = 2 ** self.up_down_times * self.filters_base
        for i in range(self.conv_times):
            x = conv3d(x, filters, self.kernel_size, name='Bottom/ConvLayer_%d' % i, mode=conv_mode)

        # Up sampling
        for layer in range(self.up_down_times - 1, -1, -1):
            filters = 2 ** layer * self.filters_base
            if self.cf.upsample == 'UpSample':
                x = UpSampling3D(size=poolSize, name='UpSample_%d' % layer)(x)
                # Reduce the number of channels
                x = conv3d(x, filters, 1, name='ChangeCh_%d' % layer, mode='')
            else:
                x = conv3d_transpose(x, filters, (1, 2, 2), strides_=stride, name='TransposeConv_%d' % layer,
                                     mode='')
            x = Concatenate(axis=-1, name='Up_%d/SkipConnection' % layer)([x, skip_connection[layer]])

            for i in range(self.conv_times):
                x = conv3d(x, filters, self.kernel_size, name='Up_%d/ConvLayer_%d' % (layer, i), mode=conv_mode)

        # Output layer
        model_output = conv3d(x, self.cf.channel_size, 1, name='UNet3D/Output', mode='')

        if self.cf.learn_residual:
            model_output = Subtract(name='Residual/Output')([model_input, model_output])

        self.model = Model(inputs=model_input,
                           outputs=model_output,
                           name='UNet_3D')

        # Compile
        optimizer = Optimizer().make(self.cf.optimizer, self.cf.learning_rate_base)
        compile_weights = [1]
        if self.cf.loss == 'L2':
            compile_losses = ['MSE']
        elif self.cf.loss == 'L1':
            compile_losses = ['MAE']
        compile_metrics = [structural_similarity, peak_snr]
        self.model.compile(optimizer=optimizer,
                           loss=compile_losses,
                           loss_weights=compile_weights,
                           metrics=compile_metrics)

        print('\n > Build UNet_3D model successfully...')
        self.model.summary()
        print([var.name for var in self.model.trainable_variables])

        return self.model


'''
UNet with residual block
https://medium.com/@nishanksingla/unet-with-resblock-for-semantic-segmentation-dd1766b4ff66
'''
class ResUNet_3D:
    def __init__(self, cf):
        self.model = None
        self.cf = cf
        self.kernel_size = cf.kernel_size       # kernal size along each dimension, 3
        self.filters_base = cf.filter_num_base  # original filter number,
        self.up_down_times = cf.up_down_times   # number of levels - 1, 4
        self.block_times = cf.conv_times         # number of convs each level, 1

    def ResBlock_3D(self, input_, out_ch, mode='', bottleneck=False,  name='ResBlock'):
        in_ch = input_.shape[-1]

        if bottleneck:
            x = conv3d(input_, in_ch, 1, name=name + '/L1', mode=mode+'R')
            x = conv3d(x, in_ch, 3, name=name + '/L2', mode=mode+'R')
            x = conv3d(x, out_ch, 1, name=name + '/L3', mode=mode+'')
        else:
            x = conv3d(input_, out_ch, 3, name=name + '/L1', mode=mode+'R')
            x = conv3d(x, out_ch, 3, name=name + '/L2', mode=mode+'')

        if in_ch == out_ch:
            shortcut = input_
        else:
            shortcut = conv3d(input_, out_ch, 1, name=name+'/Shortcut', mode=mode+'')

        x = Add(name=name+'/Output')([x, shortcut])
        x = Activation('relu')(x)
        return x

    def make(self):
        poolSize = (2, 2, 2)
        stride = (2, 2, 2)

        conv_mode = 'F'
        if self.cf.weight_decay:
            conv_mode += 'K'

        if self.cf.norm == 'Batch':
            conv_mode += 'B'
        elif self.cf.norm == 'Layer':
            conv_mode += '1'
        elif self.cf.norm == 'Group':
            conv_mode += '2'
        elif self.cf.norm == 'Instance':
            conv_mode += '3'
        else:
            pass

        skip_connection = []
        # Input layer
        model_input = Input(shape=self.cf.model_shape, name='ResUNet3D/Input')
        x = model_input

        # Down sampling
        for layer in range(self.up_down_times):
            filters = (2 ** layer) * self.filters_base
            # The filter number for each convolution in one level is the same
            for i in range(self.block_times):
                x = self.ResBlock_3D(x, filters, conv_mode, self.cf.bottleneck, name='Down_%d/ResBlock_%d' % (layer, i))

            skip_connection.append(x)
            # Down-sampling
            if self.cf.downsample == 'StridedConv':
                x = conv3d(x, filters, self.kernel_size, stride, name='Down_%d/StridedConv' % layer, mode='')
            else:
                x = MaxPool3D(pool_size=poolSize, strides=stride, name='Down_%d/MaxPool3D' % layer)(x)

            if self.cf.dropout:
                x = Dropout(rate=0.2)(x)

        # Bottom layer
        filters = 2 ** self.up_down_times * self.filters_base
        for i in range(self.block_times):
            x = self.ResBlock_3D(x, filters, conv_mode, self.cf.bottleneck, name='Bottom/ResBlock_%d' % i)

        # Up sampling
        for layer in range(self.up_down_times - 1, -1, -1):
            filters = 2 ** layer * self.filters_base
            if self.cf.upsample == 'UpSample':
                x = UpSampling3D(size=poolSize, name='UpSample_%d' % layer)(x)
                # Reduce the number of channels
                x = conv3d(x, filters, 1, name='ChangeCh_%d' % layer, mode='')
            else:
                x = conv3d_transpose(x, filters, kernel_size_=poolSize, strides_=stride, name='TransposeConv_%d' % layer, mode='')

            x = Concatenate(axis=-1, name='Up_%d/SkipConnection' % layer)([x, skip_connection[layer]])
            if self.cf.dropout:
                x = Dropout(rate=0.2)(x)

            for i in range(self.block_times):
                x = self.ResBlock_3D(x, filters, conv_mode, self.cf.bottleneck, name='Up_%d/ResBlock_%d' % (layer, i))

        # Output layer
        model_output = conv3d(x, self.cf.channel_size, 1, name='ResUNet3D/Output', mode='')

        if self.cf.learn_residual:
            model_output = Subtract(name='Residual/Output')([model_input, model_output])

        self.model = Model(inputs=model_input,
                           outputs=model_output,
                           name='ResUNet3D')

        # Compile
        optimizer = Optimizer().make(self.cf.optimizer, self.cf.learning_rate_base)
        compile_weights = [1]
        if self.cf.loss == 'L2':
            compile_losses = ['MSE']
        elif self.cf.loss == 'L1':
            compile_losses = ['MAE']
        compile_metrics = [structural_similarity, peak_snr]
        self.model.compile(optimizer=optimizer,
                           loss=compile_losses,
                           loss_weights=compile_weights,
                           metrics=compile_metrics)

        print('\n > Build ResUNet_3D model successfully...')
        self.model.summary()
        print([var.name for var in self.model.trainable_variables])

        return self.model


class DnCNN_3D:
    def __init__(self, cf):
        self.model = None
        self.cf = cf
        self.filters = cf.filter_num
        self.kernel_size = cf.kernel_size
        self.net_depth = cf.net_depth  # Total depth of network

    def make(self):
        model_input = Input(shape=self.cf.model_shape, name='DnCNN_3D/Input')
        # Input layer
        x = conv3d(model_input, self.filters, self.kernel_size, name='Layer0', mode='FR')

        conv_mode = 'F'
        if self.cf.norm:
            conv_mode += 'B'
        conv_mode += 'R'

        # Hidden layers
        for i in range(self.net_depth - 2):
            x = conv3d(x, self.filters, self.kernel_size, name='Layer_%d' % (i+1), mode=conv_mode)

        # Output layer
        x = conv3d(x, self.cf.channel_size, self.kernel_size, name='Layer_last', mode='F')
        model_output = Subtract(name='DnCNN_3D/Output')([model_input, x])

        self.model = Model(inputs=model_input,
                           outputs=model_output,
                           name='DnCNN_3D')

        # Compile
        optimizer = Optimizer().make(self.cf.optimizer, self.cf.learning_rate_base)
        compile_weights = [1]

        if self.cf.loss == 'L2':
            compile_losses = ['MSE']
        elif self.cf.loss == 'L1':
            compile_losses = ['MAE']

        compile_metrics = [structural_similarity, peak_snr]
        self.model.compile(optimizer=optimizer,
                           loss=compile_losses,
                           loss_weights=compile_weights,
                           metrics=compile_metrics)

        print('\n > Build DnCNN_3D model successfully...')
        self.model.summary()
        print([var.name for var in self.model.trainable_variables])
        return self.model


class CDDN_3D:
    # Cascaded Dilated Dense Network
    def __init__(self, cf):
        self.model = None
        self.cf = cf
        self.filters = cf.filter_num
        self.kernel_size = cf.kernel_size
        self.block_num = cf.DDB_block   # Number of dilated dense blocks

    def make(self):
        model_input = Input(shape=self.cf.model_shape, name='CDDN_3D/Input')
        # Input layer
        x = conv3d(model_input, self.filters, self.kernel_size, name='InputConv1', mode='FBR')
        x = conv3d(x, self.filters, self.kernel_size, name='InputConv2', mode='FBR')
        # Hidden layers
        for i in range(self.block_num):
            x = DDB3d(x, channel=self.filters, inner_filter=32, outer_filter=self.filters, kernel_size=self.kernel_size,
                      dense_layers=3, dilate_multi=2, compression=0.5, name='DDB_%d' % (i+1))

        # Output layer
        x = conv3d(x, self.filters, self.kernel_size, name='OutputConv1', mode='FBR')
        x = conv3d(x, self.cf.channel_size, self.kernel_size, name='OutputConv2', mode='F')
        model_output = Subtract(name='CDDN_3D/Output')([model_input, x])

        self.model = Model(inputs=model_input,
                           outputs=model_output,
                           name='CDDN_3D')

        # Compile
        optimizer = Optimizer().make(self.cf.optimizer, self.cf.learning_rate_base)
        compile_weights = [1]

        if self.cf.loss == 'L2':
            compile_losses = ['MSE']
        elif self.cf.loss == 'L1':
            compile_losses = ['MAE']

        compile_metrics = [structural_similarity, peak_snr]
        self.model.compile(optimizer=optimizer,
                           loss=compile_losses,
                           loss_weights=compile_weights,
                           metrics=compile_metrics)

        print('\n > Build CDDN_3D model successfully...')
        self.model.summary()
        print([var.name for var in self.model.trainable_variables])
        return self.model