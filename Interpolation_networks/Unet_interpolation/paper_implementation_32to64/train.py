import sys
import os
import tensorflow as tf
from tensorflow import keras
from data_gen import *
from unet_model_paper import *

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tf.config.list_physical_devices('GPU'))

#Hacer que tf vaya asignando solo la cantidad de memoria requerida
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

epochs = 200
batch_size= 16

print("Creating datasets .....")
#Training dataset
train_dataset = tf.data.Dataset.from_generator(generator=train_data_generator, output_types=(tf.float32, tf.float32))
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.__iter__()
#Validation dataset
valid_dataset = tf.data.Dataset.from_generator(generator=valid_data_generator, output_types=(tf.float32, tf.float32))
valid_dataset = valid_dataset.batch(batch_size)
valid_dataset = valid_dataset.__iter__()

#a, b = next(valid_dataset)
#print(a.shape)
#print(b.shape)

print("Creating net .....")
model = UNet()
#model.summary()

def get_check_callback():
    path_check_folder = os.path.join(r'.\model_results')
    path_csv_check = os.path.join(r'.\model_results\csv_train.csv')
    #tensorboard_logs_folder = os.path.join(path_folder, r'LIDAR_super_resolution\Scripts\unet_based\checkpoints3\tensorboard_logs')
   
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path_check_folder,'best_{epoch:02d}_{val_loss:.4f}.h5'),#path_check_folder,
                                                                   save_weights_only=False,
                                                                   monitor='val_loss',
                                                                   verbose=0,
                                                                   mode='min',
                                                                   save_best_only=True)
    csv_logger = keras.callbacks.CSVLogger(path_csv_check)
    
    #tensorboard_model = keras.callbacks.TensorBoard(log_dir=tensorboard_logs_folder)
    #command = 'tensorboard --logdir=' + os.path.join(tensorboard_logs_folder, 'logs') + ' &'
    #os.system(command)     
    
    return model_checkpoint_callback, csv_logger#, tensorboard_model

def train():  
    
    #model_checkpoint_callback, csv_logger, tensorboard_model = get_check_callback()
    model_checkpoint_callback, csv_logger = get_check_callback()

    model.fit(train_dataset,
              validation_data=valid_dataset,
              validation_steps=round(len(valid_txt_files)/batch_size),
              epochs=epochs,
              steps_per_epoch=round(len(train_txt_files)/batch_size),
              verbose=1,
              #callbacks=[model_checkpoint_callback, csv_logger, tensorboard_model])
              callbacks=[model_checkpoint_callback, csv_logger])
    
    model.save(os.path.join(path_folder, r'.\model_results\final.h5'))

if __name__ == '__main__':

    # -> train network
    print("Training process begins .....")
    train()