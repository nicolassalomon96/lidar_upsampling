#Script para generar dataset con nubes de puntos interpoladas con técnica Unet
import sys
import torch
from tqdm import tqdm
from pointcloud_utils_functions_v2 import *
from data_gen_all import *
from tensorflow import keras
from config import *
import matplotlib.pyplot as plt

pointcloud_saved_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\PointPillars\dataset\unetMC_paper_implementation\training\velodyne'

############################################## RED DE INTERPOLACIÓN ENTRENADA ####################################################
model_path = r'.\model_results\best_176_0.0053.h5'
model = keras.models.load_model(model_path)

################################################ GENERACIÓN DE DATASET ######################################################
class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()
    
train_dataset = IterDataset(data_generator)

batch_size = 5
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

############################################### MONTECARLO DROPOUT ###########################################################
def MC_drop(lr_image, hr_image, model, iterate_count=50):
    test_data_input = lr_image
    this_test = np.empty([iterate_count, low_res, image_columns, channel_num], dtype=np.float32)
    test_data_prediction = np.empty([test_data_input.shape[0], high_res, image_columns, 2], dtype=np.float32) #[bz, 64, 1024, 2]
    
    for i in range(test_data_prediction.shape[0]):

        print('Processing {} th of {} images ... '.format(i, test_data_prediction.shape[0]))
        
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

################################################ CREACIÓN DEL MODELO Y PROCESADO DE DATOS ####################################
def interpolate(input, net):
    input = input.numpy()
    output = net.predict(input) #numpy.ndarray [bz, 64, 1024, 1]
    return output

pointcloud_number = 0
for i, data in enumerate(tqdm(dataloader)):
    lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity = data
    lrimg_distance = lrimg_distance.numpy()
    hrimg_distance = hrimg_distance.numpy()
    lrimg_intensity = lrimg_intensity.numpy()
    hrimg_intensity = hrimg_intensity.numpy()
    
    ###################################### PREDICCIÓN DE PIXELES DE DISTANCIA #########################################
    #outputs_distance = interpolate(lrimg_distance, distance_model)
    #_, outputs_distance = MC_drop(lrimg_distance, hrimg_distance, distance_model)

    _, outputs_distance = MC_drop(lrimg_distance, hrimg_distance, model)
    outputs_distance = torch.from_numpy(outputs_distance) #[bz, 64, 1024, 1]
    outputs_distance = outputs_distance.permute(0, 3, 1, 2).to(device) #[bz, 1, 1024, 64]
    pointcloud_gen_batch_distance = range_image_to_pointcloud_pytorch(outputs_distance * kitti_max_distance, device)

    ###################################### PREDICCIÓN DE PIXELES DE INTENSIDAD #########################################
    #outputs_intensity = interpolate(lrimg_intensity, intensity_model)
    #_, outputs_intensity = MC_drop(lrimg_intensity, hrimg_intensity, intensity_model)

    _, outputs_intensity = MC_drop(lrimg_intensity, hrimg_intensity, model)
    outputs_intensity = torch.from_numpy(outputs_intensity)
    outputs_intensity = outputs_intensity.permute(0, 3, 1, 2).to(device)

    for j, pointcloud in enumerate(pointcloud_gen_batch_distance):
        pointcloud_join = torch.hstack([pointcloud[:,:3], outputs_intensity[j,0,:,:].reshape(-1,1)])
        save_path = os.path.join(pointcloud_saved_path, f'{str(pointcloud_number).zfill(6)}.bin')
        save_bin(pointcloud_join.cpu().detach().numpy(), save_path)
        #save_path = os.path.join(pointcloud_saved_path, f'{str(pointcloud_number).zfill(6)}.ply')
        #save_ply(pointcloud_join.cpu().detach().numpy(), save_path)
        pointcloud_number += 1