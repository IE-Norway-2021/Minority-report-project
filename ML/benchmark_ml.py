# Imports needed
import os
import gc
import sys
import time
import inspect

import cv2
import tensorflow as tf
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras import backend, activations, initializers
from tensorflow.keras.regularizers import l2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.setrecursionlimit(3000)

batch_size = 4
split_value = 0.1
EPOCHS = 150
INIT_LR = 0.00001


# definition of the models for benchmark

# densenet 121 169 and 202 code
def densenet(input_shape, n_classes, filters=32, Type=121):
    # batch norm + relu + conv
    def bn_rl_conv(x, filters, kernel=(1, 1, 1), strides=(1, 1, 1)):

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv3D(filters, kernel, strides=strides, padding='same')(x)
        return x

    def dense_block(x, repetition):
        for _ in range(repetition):
            y = bn_rl_conv(x, 4 * filters)
            y = bn_rl_conv(y, filters, 3)
            x = concatenate([y, x])
        return x

    def transition_layer(x):
        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = AvgPool3D(2, strides=2, padding='same')(x)
        return x

    input = Input(input_shape)
    x = Conv3D(64, (7, 7, 7), strides=(2, 2, 2), padding='same')(input)
    x = MaxPool3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    rep_tab = [6, 12, 24, 16]
    if Type == 169:
        rep_tab = [6, 12, 32, 32]
    elif Type == 201:
        rep_tab = [6, 12, 48, 32]

    for repetition in rep_tab:
        d = dense_block(x, repetition)
        x = transition_layer(d)
        x = GlobalAveragePooling3D()(d)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output, name=f'densenet-{Type}')
    return model


# inception res net v2 code
def inception_res_net_v2(input_shape, n_classes):
    def conv3d(x, numfilt, filtsz, strides=1, pad='same', act=True, name=None):
        x = Conv3D(numfilt, filtsz, strides, padding=pad, data_format='channels_last', use_bias=False,
                   name=name + 'conv3D')(x)
        x = BatchNormalization(axis=3, scale=False, name=name + 'conv3D' + 'bn')(x)
        if act:
            x = Activation('relu', name=name + 'conv3D' + 'act')(x)
        return x

    def incresA(x, scale, name=None):
        pad = 'same'
        branch0 = conv3d(x, 32, 1, 1, pad, True, name=name + 'b0')
        branch1 = conv3d(x, 32, 1, 1, pad, True, name=name + 'b1_1')
        branch1 = conv3d(branch1, 32, 3, 1, pad, True, name=name + 'b1_2')
        branch2 = conv3d(x, 32, 1, 1, pad, True, name=name + 'b2_1')
        branch2 = conv3d(branch2, 48, 3, 1, pad, True, name=name + 'b2_2')
        branch2 = conv3d(branch2, 64, 3, 1, pad, True, name=name + 'b2_3')
        branches = [branch0, branch1, branch2]
        mixed = Concatenate(axis=3, name=name + '_concat')(branches)
        filt_exp_1x1 = conv3d(mixed, 384, 1, 1, pad, False, name=name + 'filt_exp_1x1')
        final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                           output_shape=backend.int_shape(x)[1:],
                           arguments={'scale': scale},
                           name=name + 'act_scaling')([x, filt_exp_1x1])
        return final_lay

    def incresB(x, scale, name=None):
        pad = 'same'
        branch0 = conv3d(x, 192, 1, 1, pad, True, name=name + 'b0')
        branch1 = conv3d(x, 128, 1, 1, pad, True, name=name + 'b1_1')
        branch1 = conv3d(branch1, 160, [1, 7], 1, pad, True, name=name + 'b1_2')
        branch1 = conv3d(branch1, 192, [7, 1], 1, pad, True, name=name + 'b1_3')
        branches = [branch0, branch1]
        mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
        filt_exp_1x1 = conv3d(mixed, 1152, 1, 1, pad, False, name=name + 'filt_exp_1x1')
        final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                           output_shape=backend.int_shape(x)[1:],
                           arguments={'scale': scale},
                           name=name + 'act_scaling')([x, filt_exp_1x1])
        return final_lay

    def incresC(x, scale, name=None):
        pad = 'same'
        branch0 = conv3d(x, 192, 1, 1, pad, True, name=name + 'b0')
        branch1 = conv3d(x, 192, 1, 1, pad, True, name=name + 'b1_1')
        branch1 = conv3d(branch1, 224, [1, 3], 1, pad, True, name=name + 'b1_2')
        branch1 = conv3d(branch1, 256, [3, 1], 1, pad, True, name=name + 'b1_3')
        branches = [branch0, branch1]
        mixed = Concatenate(axis=3, name=name + '_mixed')(branches)
        filt_exp_1x1 = conv3d(mixed, 2048, 1, 1, pad, False, name=name + 'fin1x1')
        final_lay = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                           output_shape=backend.int_shape(x)[1:],
                           arguments={'scale': scale},
                           name=name + 'act_saling')([x, filt_exp_1x1])
        return final_lay

    img_input = Input(input_shape)

    x = conv3d(img_input, 32, 3, 2, 'valid', True, name='conv1')
    x = conv3d(x, 32, 3, 1, 'valid', True, name='conv2')
    x = conv3d(x, 64, 3, 1, 'valid', True, name='conv3')

    x_11 = MaxPooling3D(3, strides=1, padding='valid', name='stem_br_11' + '_maxpool_1')(x)
    x_12 = conv3d(x, 64, 3, 1, 'valid', True, name='stem_br_12')

    x = Concatenate(axis=3, name='stem_concat_1')([x_11, x_12])

    x_21 = conv3d(x, 64, 1, 1, 'same', True, name='stem_br_211')
    x_21 = conv3d(x_21, 64, [1, 7], 1, 'same', True, name='stem_br_212')
    x_21 = conv3d(x_21, 64, [7, 1], 1, 'same', True, name='stem_br_213')
    x_21 = conv3d(x_21, 96, 3, 1, 'valid', True, name='stem_br_214')

    x_22 = conv3d(x, 64, 1, 1, 'same', True, name='stem_br_221')
    x_22 = conv3d(x_22, 96, 3, 1, 'valid', True, name='stem_br_222')

    x = Concatenate(axis=3, name='stem_concat_2')([x_21, x_22])

    x_31 = conv3d(x, 192, 3, 1, 'valid', True, name='stem_br_31')
    x_32 = MaxPooling3D(3, strides=1, padding='valid', name='stem_br_32' + '_maxpool_2')(x)
    x = Concatenate(axis=3, name='stem_concat_3')([x_31, x_32])

    # Inception-ResNet-A modules
    x = incresA(x, 0.15, name='incresA_1')
    x = incresA(x, 0.15, name='incresA_2')
    x = incresA(x, 0.15, name='incresA_3')
    x = incresA(x, 0.15, name='incresA_4')

    # 35 × 35 to 17 × 17 reduction module.
    x_red_11 = MaxPooling3D(3, strides=2, padding='valid', name='red_maxpool_1')(x)

    x_red_12 = conv3d(x, 384, 3, 2, 'valid', True, name='x_red1_c1')

    x_red_13 = conv3d(x, 256, 1, 1, 'same', True, name='x_red1_c2_1')
    x_red_13 = conv3d(x_red_13, 256, 3, 1, 'same', True, name='x_red1_c2_2')
    x_red_13 = conv3d(x_red_13, 384, 3, 2, 'valid', True, name='x_red1_c2_3')

    x = Concatenate(axis=3, name='red_concat_1')([x_red_11, x_red_12, x_red_13])

    # Inception-ResNet-B modules
    x = incresB(x, 0.1, name='incresB_1')
    x = incresB(x, 0.1, name='incresB_2')
    x = incresB(x, 0.1, name='incresB_3')
    x = incresB(x, 0.1, name='incresB_4')
    x = incresB(x, 0.1, name='incresB_5')
    x = incresB(x, 0.1, name='incresB_6')
    x = incresB(x, 0.1, name='incresB_7')

    # 17 × 17 to 8 × 8 reduction module.
    x_red_21 = MaxPooling3D(3, strides=2, padding='valid', name='red_maxpool_2')(x)

    x_red_22 = conv3d(x, 256, 1, 1, 'same', True, name='x_red2_c11')
    x_red_22 = conv3d(x_red_22, 384, 3, 2, 'valid', True, name='x_red2_c12')

    x_red_23 = conv3d(x, 256, 1, 1, 'same', True, name='x_red2_c21')
    x_red_23 = conv3d(x_red_23, 256, 3, 2, 'valid', True, name='x_red2_c22')

    x_red_24 = conv3d(x, 256, 1, 1, 'same', True, name='x_red2_c31')
    x_red_24 = conv3d(x_red_24, 256, 3, 1, 'same', True, name='x_red2_c32')
    x_red_24 = conv3d(x_red_24, 256, 3, 2, 'valid', True, name='x_red2_c33')

    x = Concatenate(axis=3, name='red_concat_2')([x_red_21, x_red_22, x_red_23, x_red_24])

    # Inception-ResNet-C modules
    x = incresC(x, 0.2, name='incresC_1')
    x = incresC(x, 0.2, name='incresC_2')
    x = incresC(x, 0.2, name='incresC_3')

    # TOP
    x = GlobalAveragePooling3D(data_format='channels_last')(x)
    x = Dropout(0.6)(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(img_input, x, name='inception_resnet_v2')
    return model


# resnet 50 v2 code

def res_identity(x, filters):
    # renet block where dimension doesnot change.
    # The skip connection is just simple identity conncection
    # we will have 3 blocks and then input will be added

    x_skip = x  # this will be used for addition with the residual block
    f1, f2 = filters

    # first block
    x = Conv3D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # second block # bottleneck (but size kept same with padding)
    x = Conv3D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = Conv3D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # add the input
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def res_conv(x, s, filters):
    '''
    here the input size changes'''
    x_skip = x
    f1, f2 = filters

    # first block
    x = Conv3D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
    # when s = 2 then it is like downsizing the feature map
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # second block
    x = Conv3D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block
    x = Conv3D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # shortcut
    x_skip = Conv3D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)

    # add
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x


def resnet50(input_shape, n_classes):
    input_im = Input(shape=input_shape)  # cifar 10 images size
    x = ZeroPadding3D(padding=(3, 3))(input_im)

    # 1st stage
    # here we perform maxpooling, see the figure above

    x = Conv3D(64, kernel_size=(7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)
    x = MaxPooling3D((3, 3), strides=(2, 2))(x)

    # 2nd stage
    # frm here on only conv block and identity block, no pooling

    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    # 3rd stage

    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    # 4th stage

    x = res_conv(x, s=2, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))
    x = res_identity(x, filters=(256, 1024))

    # 5th stage

    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    # ends with average pooling and dense connection

    x = AveragePooling3D((2, 2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(n_classes, activation='softmax', kernel_initializer='he_normal')(x)  # multi-class

    # define the model

    model = Model(inputs=input_im, outputs=x, name='Resnet50')

    return model


# resnet101v2 code

class Scale(Layer):
    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
        del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def identity_block(input_tensor, kernel_size, filters, stage, block):
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution3D(nb_filter1, 1, 1, name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding3D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution3D(nb_filter2, kernel_size, kernel_size,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution3D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = concatenate([x, input_tensor], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = Convolution3D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a', bias=False)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation('relu', name=conv_name_base + '2a_relu')(x)

    x = ZeroPadding3D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Convolution3D(nb_filter2, kernel_size, kernel_size,
                      name=conv_name_base + '2b', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation('relu', name=conv_name_base + '2b_relu')(x)

    x = Convolution3D(nb_filter3, 1, 1, name=conv_name_base + '2c', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Convolution3D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1', bias=False)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = concatenate([x, shortcut], mode='sum', name='res' + str(stage) + block)
    x = Activation('relu', name='res' + str(stage) + block + '_relu')(x)
    return x


def resnet101_model(input_shape):
    eps = 1.1e-5
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=input_shape, name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=input_shape, name='data')

    x = ZeroPadding3D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution3D(64, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling3D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 3):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 23):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling3D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)

    return model


# resnet 152v2 cod

class Scale152(Layer):
    def __init__(self, weights=None, axis=-1, momentum=0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale152, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale152, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def resnet152_model(input_shape):
    eps = 1.1e-5
    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(input_shape, name='data')
    else:
        bn_axis = 1
        img_input = Input(input_shape, name='data')

    x = ZeroPadding3D((3, 3), name='conv1_zeropadding')(img_input)
    x = Convolution3D(64, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale152(axis=bn_axis, name='scale_conv1')(x)
    x = Activation('relu', name='conv1_relu')(x)
    x = MaxPooling3D((3, 3), strides=(2, 2), name='pool1')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    for i in range(1, 8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + str(i))

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    for i in range(1, 36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b' + str(i))

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x_fc = AveragePooling3D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    model = Model(img_input, x_fc)
    return model


movements = np.array(['scroll_right', 'scroll_left', 'scroll_up', 'scroll_down', 'zoom_in', 'zoom_out'])
sequence_length = 40


def benchmark_ml(root, rate):
    print('Starting Image loading...')
    label_map = {label: num for num, label in enumerate(movements)}
    sequences, labels = [], []
    for movement in movements:
        for dirpath, dirnames, files in os.walk(os.path.join(root, movement)):
            sequence = []
            if len(files) != 0:
                for i in range(sequence_length):
                    if i % rate == 0:
                        img = cv2.imread(os.path.join(dirpath, '{}.png'.format(i)))
                        sequence.append(img)
            if len(sequence) > 0:
                sequences.append(sequence)
                labels.append(label_map[movement])
    print('Image loading done! Starting train set creation...')
    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_value)
    print('Train set creation done!')
    del sequences
    del X
    del y
    del labels
    gc.collect()

    # models :
    first_dim = int(sequence_length / rate)
    input_shape = (first_dim, 120, 160, 3)
    model_densnet121 = densenet(input_shape, 6)
    model_densnet169 = densenet(input_shape, 6, Type=169)
    model_densnet201 = densenet(input_shape, 6, Type=201)
    model_inception_res_net_v2 = inception_res_net_v2(input_shape, 6)

    model_resnet152v2 = resnet152_model(input_shape)
    model_resnet101v2 = resnet101_model(input_shape)

    model_resnet50v2 = resnet50(input_shape, 6)

    benchmark_models = [(model_resnet50v2, "model_resnet50v2"),
                        (model_resnet101v2, "model_resnet101v2"),
                        (model_resnet152v2, "model_resnet152v2"),
                        (model_inception_res_net_v2, "model_inception_res_net_v2"),
                        (model_densnet201, "model_densnet201"), (model_densnet169, "model_densnet169"),
                        (model_densnet121, "model_densnet121")]

    for model, name in benchmark_models:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS),
            loss='categorical_crossentropy', metrics=["accuracy"],
        )
        with open(f'{name}_model_summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        history = model.fit(X_train, y_train, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(EPOCHS)

        plt.figure(figsize=(15, 15))
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.savefig(f'output/{name}_accuracy_results.png')
        plt.clf()

        plt.figure(figsize=(15, 15))
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(f'output/{name}_loss_results.png')
        np.save(f'output/{name}_training_history.npy', history.history)
        model.save(f'output/{name}_weights.h5')
        yhat = model.predict(X_val)
        ytrue = np.argmax(y_val, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        print(multilabel_confusion_matrix(ytrue, yhat))
        np.save(f'output/{name}_confusion_matrix.npy', multilabel_confusion_matrix(ytrue, yhat))
    del X_val
    del X_train
    del y_val
    del y_train
    gc.collect()


def train_benchmark():
    print('Doing benchmark rgb...')
    benchmark_ml('video_dataset/rgb', 2)
    print('Doing benchmark depth...')
    benchmark_ml('video_dataset/depth', 2)


if __name__ == '__main__':
    rate = 4
    first_dim = int(sequence_length / rate)
    input_shape = (first_dim, 120, 160, 3)
    model_densnet121 = densenet(input_shape, 6)
    model_densnet169 = densenet(input_shape, 6, Type=169)
    model_densnet201 = densenet(input_shape, 6, Type=201)
    model_inception_res_net_v2 = inception_res_net_v2(input_shape, 6)

    model_resnet152v2 = resnet152_model(input_shape)
    model_resnet101v2 = resnet101_model(input_shape)

    model_resnet50v2 = resnet50(input_shape, 6)
    os.mkdirs("output")
    train_benchmark()
