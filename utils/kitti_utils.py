import numpy as np
import os

import sys
sys.path.append(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\otras_arquitecturas\3_pytorch_interpolation')
from pointcloud_utils_functions_v2 import *

def distance_intensity_merge(pointcloud_gen_path, range_image_intensity_path, save_folder_path):
    #Toma una nube de puntos generada, reemplaza el valor de intensidad de cada punto por su valor original y lo guarda como una nube de puntos nueva
    #pointcloud_gen_path: ruta a la nube de puntos generada
    #range_image_intensity_path: ruta a la imagen de rango de intensidad original
    #save_folder_path: ruta a la carpeta donde se guardarán los archivos
    for pointcloud_file in os.listdir(pointcloud_gen_path):
        pointcloud_gen = read_bin(os.path.join(pointcloud_gen_path, pointcloud_file))
        intensity_file_name = pointcloud_file.split(sep='.')[0]
        original_intensity_values = np.load(os.path.join(range_image_intensity_path, rf'drive_{intensity_file_name}.npy'))
        original_intensities_serial = original_intensity_values.reshape(-1,1)
        pointcloud = np.hstack([pointcloud_gen[:,:3], original_intensities_serial])       
        save_bin(pointcloud, os.path.join(save_folder_path, pointcloud_file))
    

def kitti_downsampling(velodyne_path, save_path, downsampling_factor=2):
    #Realizar un downsampling sobre una nube de puntos
    #velodyne_path: ruta donde se encuentran las nubes de puntos .bin
    #save_path: ruta donde se guardarán los archivos generados
    #downsampling_factor: factor de downsampling a aplicar

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    indexes = range(0, 64, downsampling_factor)

    for file in os.listdir(velodyne_path):
        pointcloud = read_bin(os.path.join(velodyne_path, file))
        range_image_distance = pointcloud_to_range_image(pointcloud, kind='distance', size=(64, 1024))
        range_image_intensity = pointcloud_to_range_image(pointcloud, kind='intensity', size=(64, 1024))
        lrimg_distance = range_image_distance[indexes]
        lrimg_intensity = range_image_intensity[indexes]
        pointcloud_down = range_image_to_pointcloud_with_instensity(lrimg_distance, lrimg_intensity)
        save_bin(pointcloud_down, os.path.join(save_path, file))


def kitti_range_image_gen(folder_path, save_folder_path, kind='distance'):
    #Generación de imágenes de rango de distancia o intensidad
    #folder_path: ruta a la carpeta donde se encuentran los archivos binarios
    #save_folder_path: ruta donde se guardarán las imágenes de rango
    #kind: tipo de imagen de rango a generar --> 'distance', 'intensity', 'both'
    
    for pointcloud_file in os.listdir(folder_path):
        pointcloud_kitti = read_bin(os.path.join(folder_path, pointcloud_file))   
        range_image = pointcloud_to_range_image(pointcloud_kitti, size=(64,2048), filter_ego_compensed=True, kind=kind)
        range_image_name = 'drive_' + pointcloud_file.split(sep='.')[0] + '.npy'
        np.save(os.path.join(save_folder_path, range_image_name), range_image)


def kitti_range_image_downsampling(range_image_folder, range_image_down_folder, downsampling_factor=2):
    #Realizar un downsampling de las imágenes de rango
    #range_image_folder: ruta que contiene las imágenes de rango
    #range_image_down_folder: ruta donde se guardarán las imágenes reducidas
    #downsampling_factor: factor de downsampling
    counter = 0
    indexes = range(0, 64, downsampling_factor)
    images_64ch_path = os.listdir(range_image_folder)
    for numpy_array in images_64ch_path:
        high_res_image = np.load(os.path.join(range_image_folder, numpy_array))
        low_res_image = high_res_image[indexes,:]
        np.save(os.path.join(range_image_down_folder, numpy_array), low_res_image)
        counter += 1

    print(f"Se generaron {counter} imágenes de {64 // downsampling_factor} canales")