#Script para medir el tiempo requerido para interpolar las imágenes de rango empleando mi técnica
import time
from pointcloud_utils_functions_v2 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcloud_utils_functions_v2 import *
device = "cuda" if torch.cuda.is_available() else "cpu"

kitti_pointcloud_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\velodyne\001068.bin'
kitti_image_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\image_2\001068.png'
save_model_distance_path = r'.\models_result\model_cnn1d_min_82-12_Chamfer_sinnorm_no_ego_filtered.pth'
save_model_intensity_path = r'.\models_result\model_cnn1d_min_82-12_Chamfer_sinnorm_no_ego_filtered_intensity.pth'

pointcloud = read_bin(kitti_pointcloud_path)
range_distance_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=False, size=(64,2048), kind = 'distance')
range_intensity_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=False, size=(64,2048), kind='intensity')

indexes = range(0, 64, 2)
lrimg_distance = torch.from_numpy(range_distance_image[indexes])
lrimg_intensity = torch.from_numpy(range_intensity_image[indexes])

hrimg_distance = torch.from_numpy(range_distance_image)
hrimg_intensity = torch.from_numpy(range_intensity_image)

class Approx_net(nn.Module):    
    def __init__(self, input_channels, input_features, conv_filters_1, conv_filters_2, linear_features):
        super(Approx_net, self).__init__()
        #cnn layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=conv_filters_1, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=conv_filters_1, out_channels=conv_filters_2, kernel_size=1)
        # hidden layer
        self.linear_1 = nn.Linear(in_features=conv_filters_2*input_features+1, out_features=linear_features) #Para considerar el mínimo, se suma 1 a la entrada
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

model_distance = Approx_net(input_channels=1, input_features=6, conv_filters_1=8, conv_filters_2=2, linear_features=12)
model_distance.load_state_dict(torch.load(save_model_distance_path))
model_distance.to(device)

model_intensity = Approx_net(input_channels=1, input_features=6, conv_filters_1=8, conv_filters_2=2, linear_features=12)
model_intensity.load_state_dict(torch.load(save_model_intensity_path))
model_intensity.to(device)

lrimg_distance = lrimg_distance.unsqueeze(0).unsqueeze(0)
lrimg_intensity = lrimg_intensity.unsqueeze(0).unsqueeze(0)
lrimg_distance = lrimg_distance.to(device)
lrimg_intensity = lrimg_intensity.to(device)

hrimg_distance = hrimg_distance.unsqueeze(0).unsqueeze(0)
hrimg_intensity = hrimg_intensity.unsqueeze(0).unsqueeze(0)
hrimg_distance = hrimg_distance.to(device)
hrimg_intensity = hrimg_intensity.to(device)

test_iterations = 2
total_time = []
for i in range(test_iterations):
    start = time.time()

    twindows_distance = get_windows(lrimg_distance)
    twindows_distance_stack = twindows_distance.reshape(twindows_distance.shape[0] * twindows_distance.shape[1], twindows_distance.shape[2])
    pixels_distance = model_distance(twindows_distance_stack)
    pixels_distance = pixels_distance.view(lrimg_distance.shape[0], lrimg_distance.shape[1], -1, lrimg_distance.shape[-1])

    toutputs_distance = torch.zeros_like(hrimg_distance, device=device, dtype=torch.double)
    toutputs_distance[:,:,:toutputs_distance.shape[2]:2, :toutputs_distance.shape[3]] = lrimg_distance
    toutputs_distance[:,:,1:toutputs_distance.shape[2]-1:2, :toutputs_distance.shape[3]] = pixels_distance
    toutputs_distance = torch.dstack((toutputs_distance[:,:,:-1,:], lrimg_distance[:,:,-1:,:])) #Repito la última fila     

    twindows_intensity = get_windows(lrimg_intensity)
    twindows_intensity_stack = twindows_intensity.reshape(twindows_intensity.shape[0] * twindows_intensity.shape[1], twindows_intensity.shape[2])
    pixels_intensity = model_intensity(twindows_intensity_stack)
    pixels_intensity = pixels_intensity.view(lrimg_intensity.shape[0], lrimg_intensity.shape[1], -1, lrimg_intensity.shape[-1])

    toutputs_intensity = torch.zeros_like(hrimg_intensity, device=device, dtype=torch.double)
    toutputs_intensity[:,:,:toutputs_intensity.shape[2]:2, :toutputs_intensity.shape[3]] = lrimg_intensity
    toutputs_intensity[:,:,1:toutputs_intensity.shape[2]-1:2, :toutputs_intensity.shape[3]] = pixels_intensity
    toutputs_intensity = torch.dstack((toutputs_intensity[:,:,:-1,:], lrimg_intensity[:,:,-1:,:]))

    end = time.time()
    total_time.append(end - start)

print(f'Tiempo de interpolación: {sum(total_time[1:]) / (test_iterations-1)} segundos')