import os

#Paths
path_folder = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR' #Root folder
#lr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_images_16ch_1024_filter_ego_motion') #Low resolution images path
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_images_64ch_1024_filter_ego_motion') #High resolution images path
hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti_reduced_1500')
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\carla_mio\range_image_64ch')
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti_reduced_100')
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti_carla_reduced')
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti_carla_merged\range_image_64ch_ego_motion_filtered')


#Variables
kitti_max_distance = 80.0
carla_max_distance = 120.0
kitti_carla_min_range = 3.0

train_images_percent = 0.8 #Porcentaje de imágenes para entrenamiento
upsampling_ratio = 2
high_res_height = 64
high_res_width = 1024

augment_images = True
noise_std = 0.05 #Image noise standard desviation 