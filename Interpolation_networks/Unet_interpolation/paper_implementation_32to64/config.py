import os

#Paths
path_folder = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti' #Root folder
hr_distance_folder = os.path.join(path_folder, r'range_distance_images_3d_object\training_64x1024_no_ego_filtered') #High resolution images path
hr_intensity_folder = os.path.join(path_folder, r'range_intensity_images_3d_object\training_64x1024_no_ego_filtered') #High resolution images path

train_txt_file = r'.\ImageSets\train.txt'
valid_txt_file = r'.\ImageSets\val.txt'

#Variables
kitti_max_distance = 80.0
carla_max_distance = 120.0
kitti_carla_min_range = 2.0
lidar_z_pos = 1.73 #Altura de montaje del LIDAR
lidar_z_offset = 2.0

# Unet Parameters
low_res = 32 # 8, 16, 32
high_res = 64 # 16, 32, 64
image_columns = 1024
channel_num = 1
upscaling_factor = int(high_res / low_res)