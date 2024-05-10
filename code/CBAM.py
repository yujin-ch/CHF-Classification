from keras.layers import *
import keras.backend as K
from tensorflow.python.keras.backend import conv1d
import keras
from keras.layers import LayerNormalization

IMAGE_ORDERING = 'channels_last'


def ResCBAM_block(x, ratio=16):
    identity = x
    x = CBAM_block(x, ratio=ratio)
    x = keras.layers.add([x, identity])
    x = Activation('relu')(x)
    return x


def CBAM_block(cbam_feature, ratio=16, kernel_size=7):
    cbam_feature = channel_attention(cbam_feature, ratio=ratio)
    cbam_feature = spatial_attention(cbam_feature, kernel_size=kernel_size)
    return cbam_feature


def channel_attention(input_feature, ratio):
    init = input_feature
    channel = K.int_shape(init)[-1]
    # channel = input_feature._keras_shape[-1]
    channel_shape = (1, channel)
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    avg_pool = GlobalAveragePooling1D()(input_feature)
    avg_pool = Reshape(channel_shape)(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = Dropout(rate=0.1)(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPool1D()(input_feature)
    max_pool = Reshape(channel_shape)(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = Dropout(rate=0.1)(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam = Add()([avg_pool, max_pool])
    cbam = Activation('sigmoid')(cbam)

    return multiply([init, cbam])


def spatial_attention(input_feature, kernel_size):
    # print(input_feature.shape)
    inputs = LayerNormalization()(input_feature)
    avg_pool = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(inputs)
    # print(avg_pool.shape)
    max_pool = Lambda(lambda x: K.max(x, axis=2, keepdims=True))(inputs)

    concat = Concatenate(axis=2)([avg_pool, max_pool])
    # cbam_feature = Conv1D(1, 5, strides=1, padding='same', activation='sigmoid')(concat)
    cbam_feature = Conv1D(1, kernel_size=kernel_size, strides=1, padding='same')(concat)

    cbam_feature = Activation('sigmoid')(cbam_feature)
    return multiply([input_feature, cbam_feature])
