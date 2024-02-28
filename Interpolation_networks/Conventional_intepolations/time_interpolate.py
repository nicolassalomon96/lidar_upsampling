#Script para medir el tiempo requerido para interpolar las imágenes de rango empleando técnicas convencionales
import time
import cv2
from pointcloud_utils_functions_v2 import *

kitti_pointcloud_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\velodyne\001068.bin'
kitti_image_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\image_2\001068.png'

pointcloud = read_bin(kitti_pointcloud_path)
range_distance_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=False, size=(64,2048), kind = 'distance')
range_intensity_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=False, size=(64,2048), kind='intensity')

indexes = range(0, 64, 2)
lrimg_distance = range_distance_image[indexes]
lrimg_intensity = range_intensity_image[indexes]

test_iterations = 10
total_time = []
for i in range(test_iterations):
    start = time.time()
    outputs_distance = cv2.resize(lrimg_distance, (2048,64), interpolation=cv2.INTER_LINEAR)
    outputs_intensity = cv2.resize(lrimg_intensity, (2048,64), interpolation=cv2.INTER_LINEAR)
    
    #outputs_distance = cv2.resize(lrimg_distance, (2048,64), interpolation=cv2.INTER_CUBIC)
    #outputs_intensity = cv2.resize(lrimg_intensity, (2048,64), interpolation=cv2.INTER_CUBIC)

    end = time.time()
    total_time.append(end - start)

print(f'Tiempo de interpolación: {sum(total_time[1:]) / (test_iterations-1)} segundos')