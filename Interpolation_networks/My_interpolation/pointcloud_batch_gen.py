#Script para generar dataset con nubes de puntos interpoladas con mi técnica
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pointcloud_utils_functions_v2 import *
from data_gen_all import *
device = "cuda" if torch.cuda.is_available() else "cpu"

pointcloud_saved_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\PointPillars\dataset\my_interpolation\training\velodyne'

if not os.path.exists(pointcloud_saved_path):
    os.mkdir(pointcloud_saved_path)

############################################## RED DE INTERPOLACIÓN ENTRENADA ####################################################
saved_distance_model_path = r'.\models_result\model_cnn1d_min_82-12_Chamfer_sinnorm_no_ego_filtered.pth'
saved_intensity_model_path = r'.\models_result\model_cnn1d_min_82-12_Chamfer_sinnorm_no_ego_filtered_intensity.pth'

class Approx_net(nn.Module):    
    def __init__(self, input_channels, input_features, conv_filters_1, conv_filters_2, linear_features):
        super(Approx_net, self).__init__()
        #cnn layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=conv_filters_1, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=conv_filters_1, out_channels=conv_filters_2, kernel_size=1)
        # hidden layer
        self.linear_1 = nn.Linear(in_features=conv_filters_2*input_features+1, out_features=linear_features)
        self.linear_2 = nn.Linear(in_features=linear_features, out_features=1)
        
        self.act = nn.LeakyReLU()
        self.act1 = nn.ReLU()
    
    # prediction function
    def forward(self, x):
        min_value, _ = x.min(dim=1)
        x = x.unsqueeze(1)
        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        x_ = x_.view(x.size(0), -1)
        x_ = torch.hstack((x_, min_value.unsqueeze(1)))
        x_ = self.act(self.linear_1(x_))
        y = self.act1(self.linear_2(x_))
        return y

class Approx_net_intensity(nn.Module):    
    def __init__(self, input_channels, input_features, conv_filters_1, conv_filters_2, linear_features):
        super(Approx_net_intensity, self).__init__()
        #cnn layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=conv_filters_1, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=conv_filters_1, out_channels=conv_filters_2, kernel_size=1)
        # hidden layer
        self.linear_1 = nn.Linear(in_features=conv_filters_2*input_features+1, out_features=linear_features)
        self.linear_2 = nn.Linear(in_features=linear_features, out_features=1)
        
        self.act = nn.LeakyReLU()
        self.act1 = nn.ReLU()
    
    # prediction function
    def forward(self, x):
        min_value, _ = x.min(dim=1)
        x = x.unsqueeze(1)
        x_ = self.conv1(x)
        x_ = self.conv2(x_)
        x_ = x_.view(x.size(0), -1)
        x_ = torch.hstack((x_, min_value.unsqueeze(1)))
        x_ = self.act(self.linear_1(x_))
        y = self.act1(self.linear_2(x_))
        return y

distance_model = Approx_net(input_channels=1, input_features=6, conv_filters_1=8, conv_filters_2=2, linear_features=12)
distance_model.load_state_dict(torch.load(saved_distance_model_path))
distance_model.to(device)

intensity_model = Approx_net_intensity(input_channels=1, input_features=6, conv_filters_1=8, conv_filters_2=2, linear_features=12)
intensity_model.load_state_dict(torch.load(saved_intensity_model_path))
intensity_model.to(device)
################################################ GENERACIÓN DE DATASET ######################################################
class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()
    
train_dataset = IterDataset(data_generator)

batch_size = 100
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

################################################ CREACIÓN DEL MODELO Y PROCESADO DE DATOS ####################################
def interpolate(input, mlp_net):
    def get_windows(x, upscaling_factor=2):
        kernel_size = (2,3)
        padding = (0,1)
        stride = (1,1)
        
        if upscaling_factor == 2:
            windows = F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)
            windows = windows.transpose(1, 2) #Obtener los valores de la ventana o kernel ordenados por fila, donde cada fila representa una ventana serializada
            windows = windows.to(device)
        else:
            print("ERROR: Wrong Upsampling factor")
        return windows  

    windows = get_windows(input)
    windows_stack = windows.reshape(windows.shape[0] * windows.shape[1], windows.shape[2])
    pixels = mlp_net(windows_stack)#.squeeze()

    pixels = pixels.view(input.shape[0], input.shape[1], -1, input.shape[-1])

    toutputs = torch.zeros((input.shape[0], 1, 64, high_res_width), device=device, dtype=torch.double) #(input.shape[0], 1, 64, 2048)
    toutputs[:,:,:toutputs.shape[2]:2, :toutputs.shape[3]] = input
    toutputs[:,:,1:toutputs.shape[2]-1:2, :toutputs.shape[3]] = pixels
    toutputs = torch.dstack((toutputs[:,:,:-1,:], input[:,:,-1:,:])) #Repito la última fila
    
    return toutputs

pointcloud_number = 0
for i, data in enumerate(tqdm(dataloader)):
    lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity = data
    lrimg_distance = lrimg_distance.to(device)
    lrimg_intensity = lrimg_intensity.to(device)

    ###################################### PREDICCIÓN DE PIXELES DE DISTANCIA #########################################
    outputs_distance = interpolate(lrimg_distance, distance_model)
    #pointcloud_gen_batch_distance = range_image_to_pointcloud_pytorch(outputs_distance * kitti_max_distance, device)
    pointcloud_gen_batch_distance = range_image_to_pointcloud_pytorch(outputs_distance, device)
    ###################################### PREDICCIÓN DE PIXELES DE INTENSIDAD #########################################
    outputs_intensity = interpolate(lrimg_intensity, intensity_model)

    for j, pointcloud in enumerate(pointcloud_gen_batch_distance):
        pointcloud_join = torch.hstack([pointcloud[:,:3], outputs_intensity[j,0,:,:].reshape(-1,1)])
        save_path = os.path.join(pointcloud_saved_path, f'{str(pointcloud_number).zfill(6)}.bin')
        save_bin(pointcloud_join.cpu().detach().numpy(), save_path)
        pointcloud_number += 1