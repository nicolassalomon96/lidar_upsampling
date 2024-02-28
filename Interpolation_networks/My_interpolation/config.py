import os

#Paths
path_folder = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti' #Root folder

train_txt_file = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\lidar_upsampling\Interpolation_networks\interpolation_without_CNN\pytorch_implementation\ImageSets\train.txt'
valid_txt_file = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\lidar_upsampling\Interpolation_networks\interpolation_without_CNN\pytorch_implementation\ImageSets\val.txt'

velodyne_folder_distance = os.path.join(path_folder, r'range_distance_images_3d_object\training_64x1024_no_ego_filtered')
velodyne_folder_intensity = os.path.join(path_folder, r'range_intensity_images_3d_object\training_64x1024_no_ego_filtered')  #Imagenes de rango de intensidad originales de Kitti

#Variables
kitti_max_distance = 80.0
carla_max_distance = 120.0
kitti_carla_min_range = 3.0
lidar_z_pos = 1.73 #Altura de montaje del LIDAR
lidar_z_offset = 2.0

train_images_percent = 0.8 #Porcentaje de imágenes para entrenamiento
upsampling_ratio = 2
high_res_height = 64
high_res_width = 2048