import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import *

print('*'*40)
print(f'Input image resolution: {low_res} x {image_columns} x {channel_num}')
print(f'Output image resolution: {high_res} x {image_columns} x {channel_num}')
print(f'Upscaling factor: x{upscaling_factor}')
print('*'*40)

#Clase para implementar MonteCarlo Dropout en la etapa de prueba (seteando el parametro training=False en la red neuronal) sin afectar el dropout (que debe seguir en True)
class MonteCarloDropout(keras.layers.Dropout):
    def __init__(self, rate):
        self.rate = rate
        super().__init__(self.rate)
    def call(self, inputs):
        MC_dropout = True
        return super().call(inputs, training=MC_dropout)

# Custom Loss function
def ssim_loss(y_true, y_pred):
    loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0, filter_size=3))
    return loss

# Custom Loss function
def cosine_sim_loss(y_true, y_pred):
    loss = 1 - tf.reduce_mean(-tf.keras.losses.cosine_similarity(y_true, y_pred, axis=-1))
    return loss

# Custom Loss function
def huber_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.keras.losses.huber(y_true, y_pred, delta=1.0))
    return loss


def Unet():

    #First upscaling to get a 64x1024 output
    n_up_layers = int(np.log(upscaling_factor)/np.log(2))

    def down_conv(input, filters, kernel_size=(3,3), initializer=initializer, activation=activation_func):

        x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer=initializer, activation=activation, padding='same')(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, kernel_initializer=initializer, activation=activation, padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        return x

    def up_conv(input, filters, activation=activation_func, kernel_size=(3,3), strides=(2,2)):
        x = keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, kernel_initializer=initializer,
                                            activation=activation, strides=strides, padding='same')(input)

        x = keras.layers.BatchNormalization()(x)

        return x

    input_layer = keras.layers.Input(shape=[low_res, image_columns, channel_num])

    #First upscaling to get 64x1024 images
    for layer in range(n_up_layers):
        if layer == 0:
            i1 = up_conv(input_layer, n_filters, strides=(2,1))
        else:
            i1 = up_conv(i1, n_filters, strides=(2,1))

    #Unet downscaling
    d1 = down_conv(i1, n_filters)

    d2 = keras.layers.AveragePooling2D((2,2))(d1)
    d2 = MonteCarloDropout(dropout)(d2)
    #d2 = keras.layers.Dropout(dropout)(d2, training=True)

    d2 = down_conv(d2, n_filters*2)

    d3 = keras.layers.AveragePooling2D((2,2))(d2)
    d3 = MonteCarloDropout(dropout)(d3)
    #d3 = keras.layers.Dropout(dropout)(d3, training=True)
    d3 = down_conv(d3, n_filters*4)

    d4 = keras.layers.AveragePooling2D((2,2))(d3)
    d4 = MonteCarloDropout(dropout)(d4)
    #d4 = keras.layers.Dropout(dropout)(d4, training=True)
    d4 = down_conv(d4, n_filters*8)

    #Unet bottleneck
    b1 = keras.layers.AveragePooling2D((2,2))(d4)
    b1 = MonteCarloDropout(dropout)(b1)
    #b1 = keras.layers.Dropout(dropout)(b1, training=True)
    b1 = down_conv(b1, n_filters*16)
    b1 = MonteCarloDropout(dropout)(b1) 
    #b1 = keras.layers.Dropout(dropout)(b1, training=True)

    #Unet upscaling 
    u1 = up_conv(b1, n_filters*8)

    u2 = keras.layers.concatenate([d4, u1], axis=3)
    u2 = down_conv(u2, n_filters*8)
    u2 = MonteCarloDropout(dropout)(u2)
    #u2 = keras.layers.Dropout(dropout)(u2, training=True)
    u2 = up_conv(u2, n_filters*4)

    u3 = keras.layers.concatenate([d3, u2], axis=3)
    u3 = down_conv(u3, n_filters*4)
    u3 = MonteCarloDropout(dropout)(u3)
    #u3 = keras.layers.Dropout(dropout)(u3, training=True)
    u3 = up_conv(u3, n_filters*2) 

    u4 = keras.layers.concatenate([d2, u3], axis=3)
    u4 = down_conv(u4, n_filters*2)
    u4 = MonteCarloDropout(dropout)(u4)
    #u4 = keras.layers.Dropout(dropout)(u4, training=True)
    u4 = up_conv(u4, n_filters) 

    o1 = keras.layers.concatenate([d1, u4], axis=3)
    o1 = down_conv(o1, n_filters)

    output = keras.layers.Conv2D(1, (1,1), activation='sigmoid')(o1)

    mainModel = keras.Model(inputs=input_layer, outputs=output)

    tripleOut = keras.layers.Concatenate()([mainModel.output,mainModel.output,mainModel.output])




    
    # Loss Model
    selectedLayers = [1,2,3,4,9,10,11,12,17,18] #for instance

    ### Create Loss Model (VGG16) ###
    vgg = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(high_res, image_columns, 3))
    selectedOutputs = [vgg.layers[i].output for i in selectedLayers]
    lossModel = keras.Model(inputs=vgg.inputs, outputs=selectedOutputs)

    for layer in lossModel.layers:
        layer.trainable = False

    #a list with the output tensors for each selected layer:
    #selectedOutputs = [lossModel.layers[i].output for i in selectedLayers] #[lossModel.get_layer(name).output for name in selectedLayers]

        
    #a new model that has multiple outputs:
    #lossModel = keras.Model(lossModel.inputs,selectedOutputs)

    lossModelOutputs = lossModel(tripleOut) #or mainModel.output if not using tripeOut

    fullModel = keras.Model(mainModel.input, lossModelOutputs)

    fullModel.compile(optimizer=keras.optimizers.Adam(learning_rate=adam_lr), loss='mae')

    #if the line above doesn't work due to a type problem, make a list with lossModelOutputs:
    #lossModelOutputs = [lossModelOutputs[i] for i in range(len(selectedLayers))]

    return lossModel, fullModel
    

