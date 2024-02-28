import os
import numpy as np
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2DTranspose, Dropout, AveragePooling2D
from tensorflow.keras.layers import Input, Activation
from tensorflow.keras.layers import BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from config import *


def UNet():

    def conv_block(input, filters=64, kernel_size=(3,3)):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(input)
        x = BatchNormalization()(x)
        x = Activation(act_func)(x)
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', kernel_initializer=kernel_init)(x)
        x = BatchNormalization()(x)
        x = Activation(act_func)(x)
        return x

    def up_block(input, filters=64, kernel_size=(3,3), strides=(1,1)):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding='same', strides=strides, kernel_initializer=kernel_init)(input)
        x = BatchNormalization()(x)
        x = Activation(act_func)(x)
        return x

    filters = 64
    dropout_rate = 0.25
    act_func = 'relu'
    kernel_init = 'he_normal'

    inputs = Input((low_res, image_columns, channel_num))

    # upscailing
    x0 = inputs
    for _ in range(int(np.log(upscaling_factor) / np.log(2))):
        x0 = up_block(x0, filters, strides=(2,1))

    x1 = conv_block(x0, filters)

    x2 = AveragePooling2D((2,2))(x1)
    x2 = Dropout(dropout_rate)(x2, training=True)
    x2 = conv_block(x2, filters*2)
     
    x3 = AveragePooling2D((2,2))(x2)
    x3 = Dropout(dropout_rate)(x3, training=True)
    x3 = conv_block(x3, filters*4)
     
    x4 = AveragePooling2D((2,2))(x3)
    x4 = Dropout(dropout_rate)(x4, training=True)
    x4 = conv_block(x4, filters*8)
     
    y4 = AveragePooling2D((2,2))(x4)
    y4 = Dropout(dropout_rate)(y4, training=True)
    y4 = conv_block(y4, filters*16)
    y4 = Dropout(dropout_rate)(y4, training=True)
    y4 = up_block(y4, filters*8, strides=(2,2))
 
    y3 = concatenate([x4, y4], axis=3)
    y3 = conv_block(y3, filters*8)
    y3 = Dropout(dropout_rate)(y3, training=True)
    y3 = up_block(y3, filters*4, strides=(2,2))
 
    y2 = concatenate([x3, y3], axis=3)
    y2 = conv_block(y2, filters*4)
    y2 = Dropout(dropout_rate)(y2, training=True)
    y2 = up_block(y2, filters*2, strides=(2,2))
 
    y1 = concatenate([x2, y2], axis=3)
    y1 = conv_block(y1, filters*2)
    y1 = Dropout(dropout_rate)(y1, training=True)
    y1 = up_block(y1, filters, strides=(2,2))
 
    y0 = concatenate([x1, y1], axis=3)
    y0 = conv_block(y0, filters)

    outputs = Conv2D(1, (1, 1), activation=act_func)(y0)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0001, decay=0.00001),
        loss='mae',
        metrics=[tensorflow.keras.metrics.Accuracy()]
    )

    #model.summary()

    return model