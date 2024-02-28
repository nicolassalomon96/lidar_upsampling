#Generadores de imágenes de rango del dataset completo (sin train - val split)
import sys
import torch
from pointcloud_utils_functions_v2 import *
device = "cuda" if torch.cuda.is_available() else "cpu"

distance_files = os.listdir(velodyne_folder_distance)
intensity_files = os.listdir(velodyne_folder_intensity)

def data_augmentation(lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity):

    #Replace all sub-zero and upper max values because it is impossible in range images
    lrimg_distance[lrimg_distance < kitti_carla_min_range] = 0.0
    hrimg_distance[hrimg_distance < kitti_carla_min_range] = 0.0

    lrimg_distance[lrimg_distance > kitti_max_distance] = 0.0
    hrimg_distance[hrimg_distance > kitti_max_distance] = 0.0

    #lrimg_distance = lrimg_distance * (1./kitti_max_distance)
    #hrimg_distance = hrimg_distance *(1./kitti_max_distance)

    #Replace all sub-zero intensity by zero because it is impossible in range intensity images
    lrimg_intensity[lrimg_intensity < 0.0] = 0.0
    hrimg_intensity[hrimg_intensity < 0.0] = 0.0

    lrimg_distance = np.expand_dims(lrimg_distance, axis=0)
    hrimg_distance = np.expand_dims(hrimg_distance, axis=0)
    lrimg_intensity = np.expand_dims(lrimg_intensity, axis=0)
    hrimg_intensity = np.expand_dims(hrimg_intensity, axis=0)

    return lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity

def data_generator():
      
    for _, file in enumerate(distance_files):
        #Get random images
        train_path_distance = os.path.join(velodyne_folder_distance, file)
        train_path_intensity = os.path.join(velodyne_folder_intensity, file) 

        if train_path_distance[-3:] == 'npy':
            hrimg_distance = np.load(train_path_distance)
            hrimg_intensity = np.load(train_path_intensity)
        else:
            print("Wrong pointcloud filepath")
        
        indexes = range(0, high_res_height, upsampling_ratio)

        lrimg_distance = hrimg_distance[indexes]
        lrimg_intensity = hrimg_intensity[indexes]

        lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity = data_augmentation(lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity)
        yield (lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity)
        #lrimg_distance, hrimg_distance = data_augmentation(lrimg_distance, hrimg_distance)
        #yield (lrimg_distance, hrimg_distance)