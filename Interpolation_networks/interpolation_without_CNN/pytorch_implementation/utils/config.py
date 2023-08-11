import os

#Paths
path_folder = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR' #Root folder
#lr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_images_16ch_1024_filter_ego_motion') #Low resolution images path
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_images_64ch_1024_filter_ego_motion') #High resolution images path
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_images_64ch_2048_filter_ego_motion')
hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_distance_images_3d_object\testing_64x2048') #El entrenamiento lo realicé con 64x1024
hr_intensity_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_intensity_images_3d_object\testing_64x2048')
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_dist_int_64ch_1024_filter_ego_motion')
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti_reduced_1500')
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti_reduced_1500_distance_intensity')
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\carla_mio\range_image_64ch')
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti_reduced_100')
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti_carla_reduced')
#hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti_carla_merged\range_image_64ch_ego_motion_filtered')


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