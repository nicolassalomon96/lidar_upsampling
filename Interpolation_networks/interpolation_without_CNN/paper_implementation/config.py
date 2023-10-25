import os

#Paths
path_folder = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti' #Root folder

velodyne_folder = os.path.join(path_folder, r'range_distance_images_3d_object\training_64x2048') #El entrenamiento lo realicé con 64x1024
velodyne_folder_1image = os.path.join(path_folder, r'range_distance_images_3d_object\training_64x1024_1image') #El entrenamiento lo realicé con 64x1024
labels_folder = os.path.join(path_folder, r'kitti_3d_object\training\label_2')

#Variables
kitti_max_distance = 80.0
carla_max_distance = 120.0
kitti_carla_min_range = 3.0
lidar_z_pos = 1.73 #Altura de montaje del LIDAR
lidar_z_offset = 2.0

train_images_percent = 0.8 #Porcentaje de imágenes para entrenamiento
upsampling_ratio = 2
high_res_height = 64
high_res_width = 1024

noise_std = 0.05 #Image noise standard desviation 