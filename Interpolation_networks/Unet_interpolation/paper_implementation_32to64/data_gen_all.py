#Generadores de imágenes de rango del dataset completo (sin train - val split)
import sys
import torch
from pointcloud_utils_functions_v2 import *
device = "cuda" if torch.cuda.is_available() else "cpu"

distance_files = os.listdir(hr_distance_folder)
intensity_files = os.listdir(hr_intensity_folder)

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
        
        lrimg = np.expand_dims(lrimg, axis=-1)
        hrimg = np.expand_dims(hrimg, axis=-1)

    return lrimg, hrimg

def data_generator():
   
    indexes = range(0, high_res, upscaling_factor)
    for _, file in enumerate(distance_files):
        #Get random images
        path_distance = os.path.join(hr_distance_folder, file) #Formato original de kitti: drive_000000.npy    
        train_path_intensity = os.path.join(hr_intensity_folder, file)

        if path_distance[-3:] == 'npy':
            hrimg_distance = np.load(path_distance)
        if train_path_intensity[-3:] == 'npy':
            hrimg_intensity = np.load(train_path_intensity)
        else:
            print("Wrong pointcloud filepath")        

        lrimg_distance = hrimg_distance[indexes]
        lrimg_intensity = hrimg_intensity[indexes]

        lrimg_distance, hrimg_distance = data_augmentation(lrimg_distance, hrimg_distance, image='distance')
        #yield (lrimg_distance, hrimg_distance)
        
        lrimg_intensity, hrimg_intensity = data_augmentation(lrimg_intensity, hrimg_intensity, image='intensity')
        #yield(lrimg_intensity, hrimg_intensity)

        yield(lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity)