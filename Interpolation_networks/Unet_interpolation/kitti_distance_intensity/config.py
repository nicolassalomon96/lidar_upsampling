import os

#Paths
path_folder = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR' #Root folder
lr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_distance_intensity_16ch') #Low resolution images path
hr_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_images_64ch') #High resolution images path
#lr_test_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti_carla_merged\range_image_16ch_test') #Low resolution images path
#hr_test_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti_carla_merged\range_image_64ch_test') #High resolution images path

#lr_intensity_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_images_intensity_16ch') #Low resolution images path
#hr_intensity_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_images_intensity_64ch') #High resolution images path
#lr_intensity_test_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_images_intensity_16ch_test') #Low resolution images path
#hr_intensity_test_folder = os.path.join(path_folder, r'Datasets LIDAR\kitti\range_images_intensity_64ch_test') #High resolution images path


#Variables
kitti_max_distance = 120
carla_max_distance = 120

train_images_percent = 0.9 #Porcentaje de imágenes para entrenamiento
noise_std = 0.05 #Image noise standard desviation 


# Unet Parameters
low_res = 16 # 8, 16, 32
high_res = 64 # 16, 32, 64
image_columns = 1024
channel_num = 2
upscaling_factor = int(high_res / low_res)

#Model Parameters
n_filters = 64
activation_func = 'relu'
output_fn = 'sigmoid'
dropout = 0.2
initializer = 'he_normal'
adam_lr = 0.001