#Script para medir el tiempo requerido para interpolar las imágenes de rango empleando técnica Unet
from pointcloud_utils_functions_v2 import *
from data_gen_all import *
from tensorflow import keras
from config import *
import time
import sys

model_path = r'.\model_results\best_176_0.0053.h5'
model = keras.models.load_model(model_path)

kitti_pointcloud_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\velodyne\001068.bin'
kitti_image_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\image_2\001068.png'

pointcloud = read_bin(kitti_pointcloud_path)
range_distance_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=False, size=(64,1024), kind = 'distance')
range_intensity_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=False, size=(64,1024), kind='intensity')

indexes = range(0, 64, 2)
lrimg_distance = range_distance_image[indexes]
lrimg_intensity = range_intensity_image[indexes]

def MC_drop(lr_image, hr_image, model, iterate_count=50):
    test_data_input = lr_image
    this_test = np.empty([iterate_count, low_res, image_columns, channel_num], dtype=np.float32)
    test_data_prediction = np.empty([test_data_input.shape[0], high_res, image_columns, 2], dtype=np.float32) #[bz, 64, 1024, 2]
    
    for i in range(test_data_prediction.shape[0]):

        #print('Processing {} th of {} images ... '.format(i, test_data_prediction.shape[0]))
        
        for j in range(iterate_count):
            this_test[j] = test_data_input[i]

        this_prediction = model.predict(this_test, verbose=1)

        this_prediction_mean = np.mean(this_prediction, axis=0)
        this_prediction_var = np.std(this_prediction, axis=0)
        test_data_prediction[i,:,:,0:1] = this_prediction_mean
        test_data_prediction[i,:,:,1:2] = this_prediction_var

    low_res_index = range(0, high_res, upscaling_factor)
    test_data_prediction[:,low_res_index,:,0:1] = hr_image[:,low_res_index,:,0:1]
    test_data_prediction_reduced = np.copy(test_data_prediction[:,:,:,0:1])
    
    # remove noise
    if len(test_data_prediction.shape) == 4 and test_data_prediction.shape[-1] == 2:
        noiseLabels = test_data_prediction[:,:,:,1:2]
        test_data_prediction_reduced[noiseLabels > test_data_prediction_reduced * 0.03] = 0 # after noise removal
        test_data_prediction_reduced[:,low_res_index] = hr_image[:,low_res_index]

    test_data_prediction[:,:,:,1:2] = None
    return test_data_prediction, test_data_prediction_reduced


test_iterations = 10
total_time = []
for i in range(test_iterations):
    start = time.time()
    _, outputs_distance = MC_drop(lrimg_distance.reshape(1, 32, 1024, 1), range_distance_image.reshape(1, 64, 1024, 1), model)
    _, outputs_intensity = MC_drop(lrimg_intensity.reshape(1, 32, 1024, 1), range_intensity_image.reshape(1, 64, 1024, 1), model)
    end = time.time()
    total_time.append(end - start)

print(f'Tiempo de interpolación: {sum(total_time[1:]) / (test_iterations-1)} segundos')