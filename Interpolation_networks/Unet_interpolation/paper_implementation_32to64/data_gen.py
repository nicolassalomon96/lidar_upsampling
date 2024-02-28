#Generadores de imágenes de rango según train - val split
import numpy as np
import os
from config import *
from pointcloud_utils_functions_v2 import *

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(train_txt_file, 'r') as archivo:
    # Lee todas las líneas del archivo y las guarda en una lista
    lineas = archivo.readlines()

train_txt_files = [linea.rstrip('\n') for linea in lineas]

with open(valid_txt_file, 'r') as archivo:
    # Lee todas las líneas del archivo y las guarda en una lista
    lineas = archivo.readlines()

valid_txt_files = [linea.rstrip('\n') for linea in lineas]

#def data_augmentation(lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity):
def data_augmentation(lrimg, hrimg, image='distance'):

    if image == 'distance':
        #Replace all sub-zero and upper max values because it is impossible in range images
        lrimg[lrimg < kitti_carla_min_range] = 0.0
        hrimg[hrimg < kitti_carla_min_range] = 0.0

        lrimg[lrimg > kitti_max_distance] = 0.0
        hrimg[hrimg > kitti_max_distance] = 0.0

        lrimg = np.expand_dims(lrimg, axis=-1)
        hrimg = np.expand_dims(hrimg, axis=-1)

        lrimg = lrimg * (1./kitti_max_distance)
        hrimg = hrimg *(1./kitti_max_distance)

    if image == 'intensity':
        #Replace all sub-zero intensity by zero because it is impossible in range intensity images
        lrimg[lrimg < 0.0] = 0.0
        hrimg[hrimg < 0.0] = 0.0
        
        lrimg = np.expand_dims(lrimg, axis=0)
        hrimg = np.expand_dims(hrimg, axis=0)

    #return lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity
    return lrimg, hrimg

def train_data_generator():
   
    indexes = range(0, high_res, upscaling_factor)
    while True:
        for _, file in enumerate(train_txt_files):
            #Get random images
            train_path_distance = os.path.join(hr_distance_folder, f'drive_{file}.npy') #Formato original de kitti: drive_000000.npy    
            #train_path_intensity = os.path.join(hr_intensity_folder, f'drive_{file}.npy')

            if train_path_distance[-3:] == 'npy':
                hrimg_distance = np.load(train_path_distance)
            #if train_path_intensity[-3:] == 'npy':
            #    hrimg_intensity = np.load(train_path_intensity)
            else:
                print("Wrong pointcloud filepath")        

            lrimg_distance = hrimg_distance[indexes]
            #lrimg_intensity = hrimg_intensity[indexes]

            lrimg_distance, hrimg_distance = data_augmentation(lrimg_distance, hrimg_distance, image='distance')
            yield (lrimg_distance, hrimg_distance)
            
            #lrimg_intensity, hrimg_intensity = data_augmentation(lrimg_intensity, hrimg_intensity, image='intensity')
            #yield(lrimg_intensity, hrimg_intensity)

def valid_data_generator():
    
    indexes = range(0, high_res, upscaling_factor)
    while True:
        for _, file in enumerate(valid_txt_files):

            #Get random images
            valid_path_distance = os.path.join(hr_distance_folder, f'drive_{file}.npy') 
            #valid_path_intensity = os.path.join(velodyne_folder_intensity, f'drive_{file}.npy')

            if valid_path_distance[-3:] == 'npy':
                hrimg_distance = np.load(valid_path_distance)
            #if valid_path_intensity[-3:] == 'npy':
            #    hrimg_intensity = np.load(valid_path_intensity)
            else:
                print("Wrong pointcloud filepath")
                
            lrimg_distance = hrimg_distance[indexes]
            #lrimg_intensity = hrimg_intensity[indexes]

            #lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity = data_augmentation(lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity)
            #yield (lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity)

            lrimg_distance, hrimg_distance = data_augmentation(lrimg_distance, hrimg_distance, image='distance')
            yield (lrimg_distance, hrimg_distance)

            #lrimg_intensity, hrimg_intensity = data_augmentation(lrimg_intensity, hrimg_intensity, image='intensity')
            #yield (lrimg_intensity, hrimg_intensity)