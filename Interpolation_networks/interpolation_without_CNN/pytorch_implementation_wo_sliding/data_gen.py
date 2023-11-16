import sys
import torch
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
def data_augmentation(lrimg_distance, hrimg_distance):

    #Replace all sub-zero and upper max values because it is impossible in range images
    lrimg_distance[lrimg_distance < kitti_carla_min_range] = 0.0
    hrimg_distance[hrimg_distance < kitti_carla_min_range] = 0.0

    lrimg_distance[lrimg_distance > kitti_max_distance] = 0.0
    hrimg_distance[hrimg_distance > kitti_max_distance] = 0.0

    lrimg_distance = lrimg_distance * (1./kitti_max_distance)
    hrimg_distance = hrimg_distance *(1./kitti_max_distance)

    #Replace all sub-zero intensity by zero because it is impossible in range intensity images
    #lrimg_intensity[lrimg_intensity < 0.0] = 0.0
    #hrimg_intensity[hrimg_intensity < 0.0] = 0.0

    lrimg_distance = np.expand_dims(lrimg_distance, axis=0)
    hrimg_distance = np.expand_dims(hrimg_distance, axis=0)
    #lrimg_intensity = np.expand_dims(lrimg_intensity, axis=0)
    #hrimg_intensity = np.expand_dims(hrimg_intensity, axis=0)

    #return lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity
    return lrimg_distance, hrimg_distance

def train_data_generator():
   
    #train_txt_files = os.listdir(train_txt_file)
    #velodyne_name_intensity = os.listdir(velodyne_folder_intensity)
    
    for _, file in enumerate(train_txt_files):
        #Get random images
        train_path_distance = os.path.join(velodyne_folder_distance, f'{file}.npy')
        #train_path_distance = os.path.join(velodyne_folder_distance, f'drive_{file}.npy') #Formato original de kitti: drive_000000.npy
        #train_path_intensity = os.path.join(velodyne_folder_intensity, url) 

        if train_path_distance[-3:] == 'npy':
            hrimg_distance = np.load(train_path_distance)
            #hrimg_intensity = np.load(train_path_intensity)
        else:
            print("Wrong pointcloud filepath")
        
        indexes = range(0, high_res_height, upsampling_ratio)

        lrimg_distance = hrimg_distance[indexes]
        #lrimg_intensity = hrimg_intensity[indexes]

        #lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity = data_augmentation(lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity)
        #yield (lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity)
        lrimg_distance, hrimg_distance = data_augmentation(lrimg_distance, hrimg_distance)
        yield (lrimg_distance, hrimg_distance)

def valid_data_generator():

    #valid_txt_files = os.listdir(valid_txt_file)
    #velodyne_name_intensity = os.listdir(velodyne_folder_intensity)
    
    for _, file in enumerate(valid_txt_files):

        #Get random images
        valid_path_distance = os.path.join(velodyne_folder_distance, f'{file}.npy')
        #valid_path_distance = os.path.join(velodyne_folder_distance, f'drive_{file}.npy') 
        #train_path_intensity = os.path.join(velodyne_folder_intensity, url) 

        if valid_path_distance[-3:] == 'npy':
            hrimg_distance = np.load(valid_path_distance)
            #hrimg_intensity = np.load(train_path_intensity)
        else:
            print("Wrong pointcloud filepath")
        
        indexes = range(0, high_res_height, upsampling_ratio)
        lrimg_distance = hrimg_distance[indexes]
        #lrimg_intensity = hrimg_intensity[indexes]

        #lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity = data_augmentation(lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity)
        #yield (lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity)
        lrimg_distance, hrimg_distance = data_augmentation(lrimg_distance, hrimg_distance)
        yield (lrimg_distance, hrimg_distance)

'''
def data_augmentation_distance(lrimg, hrimg, dataset=None):

    #Replace all sub-zero and upper max values because it is impossible in range images
    lrimg[lrimg < kitti_carla_min_range] = 0.0
    hrimg[hrimg < kitti_carla_min_range] = 0.0

    if dataset == 'kitti':
        lrimg[lrimg > kitti_max_distance] = 0.0
        hrimg[hrimg > kitti_max_distance] = 0.0

        #lrimg = np.array(lrimg * (1./kitti_max_distance), dtype=np.float32)
        #hrimg = np.array(hrimg * (1./kitti_max_distance), dtype=np.float32)
    
    elif dataset == 'carla':
        lrimg[lrimg > carla_max_distance] = 0.0
        hrimg[hrimg > carla_max_distance] = 0.0

        lrimg = np.array(lrimg * (1./carla_max_distance), dtype=np.float32)
        hrimg = np.array(hrimg * (1./carla_max_distance), dtype=np.float32)

    lrimg = np.expand_dims(lrimg, axis=0)
    hrimg = np.expand_dims(hrimg, axis=0)
    return lrimg, hrimg

def data_augmentation_intensity(lrimg, hrimg, dataset=None):

    #Replace all sub-zero intensity by zero because it is impossible in range intensity images
    lrimg[lrimg < 0.0] = 0.0
    hrimg[hrimg < 0.0] = 0.0

    lrimg = np.expand_dims(lrimg, axis=0)
    hrimg = np.expand_dims(hrimg, axis=0)
    return lrimg, hrimg

def distance_data_generator():
    velodyne_name = os.listdir(velodyne_folder_distance)
    for _, url in enumerate(velodyne_name):
        if url[0:5] == 'drive':
            dataset = 'kitti'
        if url[0:4] == 'Town':
            dataset = 'carla'

        #Get random images
        train_path = os.path.join(velodyne_folder_distance, url) #hr_folder + '\\' + url

        if train_path[-3:] == 'npy':
            hrimg = np.load(train_path)
        elif train_path[-3:] == 'tif':
            hrimg = io.imread(train_path)
        else:
            print("Wrong pointcloud filepath")
        
        indexes = range(0, high_res_height, upsampling_ratio)
        lrimg = hrimg[indexes]
        lrimg, hrimg = data_augmentation_distance(lrimg, hrimg, dataset)

        yield (lrimg, hrimg)

def intensity_data_generator():
    velodyne_name = os.listdir(velodyne_folder_intensity)
    for _, url in enumerate(velodyne_name):
        if url[0:5] == 'drive':
            dataset = 'kitti'
        if url[0:4] == 'Town':
            dataset = 'carla'

        #Get random images
        train_path = os.path.join(velodyne_folder_intensity, url) #hr_folder + '\\' + url

        if train_path[-3:] == 'npy':
            hrimg = np.load(train_path)
        elif train_path[-3:] == 'tif':
            hrimg = io.imread(train_path)
        else:
            print("Wrong pointcloud filepath")
        
        indexes = range(0, high_res_height, upsampling_ratio)
        lrimg = hrimg[indexes]
        lrimg, hrimg = data_augmentation_intensity(lrimg, hrimg, dataset)

        yield (lrimg, hrimg)
'''