# -*- coding: utf-8 -*-

"""
===========================================================================================================

Created on Thu Aug 10 2023
Latest update: 08-10-23 

@author: AnhND
===========================================================================================================
"""


import tensorflow as tf


from tensorflow import keras
from keras.models import Model
from keras.layers import Dropout, Conv2D, MaxPool2D, Input, Conv2DTranspose, concatenate, BatchNormalization







def block_conv(inputs, filters, kernel_size, strides,
               padding, activation, kernel_initializer, rate, data_format):

    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        data_format=data_format,
        kernel_initializer=kernel_initializer)(inputs)

    x = BatchNormalization()(x)

    x = Dropout(rate)(x)

    x = Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        data_format=data_format,
        kernel_initializer=kernel_initializer)(x)

    x = BatchNormalization()(x)

    return x


def up_concat_block(inputs, concat_inputs, filters, kernel_size, strides,
                    padding, activation, kernel_initializer, data_format):

    x = Conv2DTranspose(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        data_format=data_format,
        kernel_initializer=kernel_initializer)(inputs)

    x = concatenate([concat_inputs, x], axis=3)

    return x


def unet_2D(input_shape, filters_list, kernel_size, down_strides, up_strides,
            padding, activation, final_activation, kernel_initializer, rate,
            data_format='channels_last'):

    inputs = Input(input_shape)

    # ---------------------------Down levels-------------------------------

    # Level 0
    conv0_down = block_conv(inputs, filters_list[0], kernel_size, down_strides,
                            padding, activation, kernel_initializer, rate, data_format)

    pool0 = MaxPool2D(pool_size=(2, 2), padding='same',
                      data_format=data_format)(conv0_down)

    # Level 1
    conv1_down = block_conv(pool0, filters_list[1], kernel_size, down_strides,
                            padding, activation, kernel_initializer, rate, data_format)

    pool1 = MaxPool2D(pool_size=(2, 2), padding='same',
                      data_format=data_format)(conv1_down)

    # Lever 2
    conv2_down = block_conv(pool1, filters_list[2], kernel_size, down_strides,
                            padding, activation, kernel_initializer, rate, data_format)

    pool2 = MaxPool2D(pool_size=(2, 2), padding='same',
                      data_format=data_format)(conv2_down)

    # Lever 3
    conv3_down = block_conv(pool2, filters_list[3], kernel_size, down_strides,
                            padding, activation, kernel_initializer, rate, data_format)

    pool3 = MaxPool2D(pool_size=(2, 2), padding='same',
                      data_format=data_format)(conv3_down)

    # -------------------------------Center lever-----------------------------------
    conv_center = block_conv(pool3, filters_list[4], kernel_size, down_strides,
                             padding, activation, kernel_initializer, rate, data_format)

    # ---------------------------Up levels-------------------------------

    # Lever 3
    conv3_up = up_concat_block(conv_center, conv3_down, filters_list[3], kernel_size,
                               up_strides, padding, activation, kernel_initializer, data_format)

    up3 = block_conv(conv3_up, filters_list[3], kernel_size, down_strides,
                     padding, activation, kernel_initializer, rate, data_format)

    # Lever 2
    conv2_up = up_concat_block(up3, conv2_down, filters_list[2], kernel_size,
                               up_strides, padding, activation, kernel_initializer, data_format)

    up2 = block_conv(conv2_up, filters_list[2], kernel_size, down_strides,
                     padding, activation, kernel_initializer, rate, data_format)

    # Lever 1
    conv1_up = up_concat_block(up2, conv1_down, filters_list[1], kernel_size,
                               up_strides, padding, activation, kernel_initializer, data_format)

    up1 = block_conv(conv1_up, filters_list[1], kernel_size, down_strides,
                     padding, activation, kernel_initializer, rate, data_format)

    # Lever 0
    conv0_up = up_concat_block(up1, conv0_down, filters_list[0], kernel_size,
                               up_strides, padding, activation, kernel_initializer, data_format)

    up0 = block_conv(conv0_up, filters_list[0], kernel_size, down_strides,
                     padding, activation, kernel_initializer, rate, data_format)

    # Output
    outputs = x = Conv2D(
        filters=1,
        kernel_size=1,
        strides=down_strides,
        padding=padding,
        activation=final_activation,
        kernel_initializer=kernel_initializer,
        data_format=data_format)(up0)

    model = Model(inputs=inputs, outputs=outputs)
    return model
