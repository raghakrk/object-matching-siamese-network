import tensorflow 
from tensorflow.keras.layers import (
  Input,
  Flatten,
  Conv2D,
  MaxPooling2D,
  BatchNormalization,
  Activation,
  add,
  Convolution2D,
  ZeroPadding2D,
  AveragePooling2D,
  Concatenate
)

from tensorflow.keras.models import Model
from tensorflow.keras.applications import resnet50, vgg16
from tensorflow.keras.regularizers import l2

DATA_FORMAT = 'channels_last'
CONCAT_AXIS = -1

def inception_module(x, params, data_format, concat_axis,
                     subsample=(1, 1), activation='relu',
                     border_mode='same', weight_decay=None):

    (branch1, branch2, branch3, branch4) = params

    if weight_decay:
        kernel_regularizer = l2(weight_decay)
        bias_regularizer = l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None

    pathway1 = Convolution2D(branch1[0], (1, 1),
                             strides=subsample,
                             activation=activation,
                             padding=border_mode,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             use_bias=False,
                             data_format=data_format)(x)

    pathway2 = Convolution2D(branch2[0], (1, 1),
                             strides=subsample,
                             activation=activation,
                             padding=border_mode,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             use_bias=False,
                             data_format=data_format)(x)
    pathway2 = Convolution2D(branch2[1], (3, 3),
                             strides=subsample,
                             activation=activation,
                             padding=border_mode,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             use_bias=False,
                             data_format=data_format)(pathway2)

    pathway3 = Convolution2D(branch3[0], (1, 1),
                             strides=subsample,
                             activation=activation,
                             padding=border_mode,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             use_bias=False,
                             data_format=data_format)(x)
    pathway3 = Convolution2D(branch3[1], (5, 5),
                             strides=subsample,
                             activation=activation,
                             padding=border_mode,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             use_bias=False,
                             data_format=data_format)(pathway3)

    pathway4 = MaxPooling2D(pool_size=(1, 1), data_format=DATA_FORMAT)(x)
    pathway4 = Convolution2D(branch4[0], (1, 1),
                             strides=subsample,
                             activation=activation,
                             padding=border_mode,
                             kernel_regularizer=kernel_regularizer,
                             bias_regularizer=bias_regularizer,
                             use_bias=False,
                             data_format=data_format)(pathway4)

    return Concatenate()([pathway1, pathway2, pathway3, pathway4])

def conv_layer(x, nb_filter, nb_row, nb_col, data_format,
               subsample=(1, 1), activation='relu',
               border_mode='same', weight_decay=None, padding=None):

    if weight_decay:
        kernel_regularizer = l2(weight_decay)
        bias_regularizer = l2(weight_decay)
    else:
        kernel_regularizer = None
        bias_regularizer = None

    x = Convolution2D(nb_filter, (nb_row, nb_col),
                      strides=subsample,
                      activation=activation,
                      padding=border_mode,
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,
                      use_bias=False,
                      data_format=data_format)(x)

    if padding:
        for i in range(padding):
            x = ZeroPadding2D(padding=(1, 1), data_format=data_format)(x)

    return x

def GoogLeNet(input_shape):
    img_input = Input(input_shape)
    x = conv_layer(img_input, nb_col=7, nb_filter=64,
                   nb_row=7, data_format=DATA_FORMAT, padding=3)
    x = MaxPooling2D(
        strides=(
            3, 3), pool_size=(
                2, 2), data_format=DATA_FORMAT)(x)

    x = conv_layer(x, nb_col=1, nb_filter=64,
                   nb_row=1, data_format=DATA_FORMAT)
    x = conv_layer(x, nb_col=3, nb_filter=192,
                   nb_row=3, data_format=DATA_FORMAT, padding=1)
    x = MaxPooling2D(
        strides=(
            3, 3), pool_size=(
                2, 2), data_format=DATA_FORMAT)(x)

    x = inception_module(x, params=[(64, ), (96, 128), (16, 32), (32, )],
                         data_format=DATA_FORMAT, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 192), (32, 96), (64, )],
                         data_format=DATA_FORMAT, concat_axis=CONCAT_AXIS)

    x = MaxPooling2D(
        strides=(
            1, 1), pool_size=(
                1, 1), data_format=DATA_FORMAT)(x)
    x = ZeroPadding2D(padding=(1, 1), data_format=DATA_FORMAT)(x)

    x = inception_module(x, params=[(192,), (96, 208), (16, 48), (64, )],
                         data_format=DATA_FORMAT, concat_axis=CONCAT_AXIS)
    # AUX 1 - Branch HERE
    x = inception_module(x, params=[(160,), (112, 224), (24, 64), (64, )],
                         data_format=DATA_FORMAT, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(128,), (128, 256), (24, 64), (64, )],
                         data_format=DATA_FORMAT, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(112,), (144, 288), (32, 64), (64, )],
                         data_format=DATA_FORMAT, concat_axis=CONCAT_AXIS)
    # AUX 2 - Branch HERE
    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                         data_format=DATA_FORMAT, concat_axis=CONCAT_AXIS)
    x = MaxPooling2D(
        strides=(
            1, 1), pool_size=(
                1, 1), data_format=DATA_FORMAT)(x)

    x = inception_module(x, params=[(256,), (160, 320), (32, 128), (128,)],
                         data_format=DATA_FORMAT, concat_axis=CONCAT_AXIS)
    x = inception_module(x, params=[(384,), (192, 384), (48, 128), (128,)],
                         data_format=DATA_FORMAT, concat_axis=CONCAT_AXIS)
    x = AveragePooling2D(strides=(1, 1), data_format=DATA_FORMAT)(x)
    x = Flatten()(x)

    return Model(img_input,x)

def resnet6(input_shape):

    input1 = Input(input_shape) 

    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(input1)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    x = Flatten()(x5)

    return Model(input1,x)

def resnet8(input_shape):

    input1 = Input(input_shape)
    x1 = Conv2D(32, (5, 5), strides=[2,2], padding='same')(input1)
    x1 = MaxPooling2D(pool_size=(3, 3), strides=[2,2])(x1)

    # First residual block
    x2 = BatchNormalization()(x1)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Conv2D(32, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x2)

    x1 = Conv2D(32, (1, 1), strides=[2,2], padding='same')(x1)
    x3 = add([x1, x2])

    # Second residual block
    x4 = BatchNormalization()(x3)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x4 = BatchNormalization()(x4)
    x4 = Activation('relu')(x4)
    x4 = Conv2D(64, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x4)

    x3 = Conv2D(64, (1, 1), strides=[2,2], padding='same')(x3)
    x5 = add([x3, x4])

    # Third residual block
    x6 = BatchNormalization()(x5)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), strides=[2,2], padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x6)

    x6 = BatchNormalization()(x6)
    x6 = Activation('relu')(x6)
    x6 = Conv2D(128, (3, 3), padding='same',
                kernel_initializer="he_normal",
                kernel_regularizer=l2(1e-4))(x6)

    x5 = Conv2D(128, (1, 1), strides=[2,2], padding='same')(x5)
    x7 = add([x5, x6])

    x = Flatten()(x7)

    return Model(input1,x) 

def resnet50_model(input_shape):
  model = resnet50.ResNet50(weights=None,include_top=False,input_shape=input_shape)
  x = Flatten()(model.output)
  baseModel = Model(inputs=model.input,outputs=x)
  #for layer in baseModel.layers:
  #  layer.trainable = False
  return baseModel

def vgg16_model(input_shape):
  model = vgg16.VGG16(weights=None, include_top=False, input_shape=input_shape)
  x = Flatten()(model.output)
  baseModel = Model(inputs=model.input,outputs=x)
  #for layer in baseModel.layers:
  #  layer.trainable = False
  return baseModel

def small_vgg(input_shape):
    input1 = Input(input_shape)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',name='block1_conv1')(input1)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same',name='block2_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',name='block3_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block4_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same',name='block5_conv1')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten()(x)

    return Model(input1,x)