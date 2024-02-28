#Script para realizar un test sobre un frame del dataset de kitti
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointcloud_utils_functions_v2 import *
kitti_max_distance = 80.0

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
        #print(f'Salida con1d-1: {x_.shape}')
        x_ = self.conv2(x_)
        #x_ = self.conv3(x_)
        #print(f'Salida con1d-2: {x_.shape}')
        x_ = x_.view(x.size(0), -1)
        x_ = torch.hstack((x_, min_value.unsqueeze(1)))
        #print(f'Entrada MLP: {x_.shape}')
        x_ = self.act(self.linear_1(x_))
        #print(f'Salida primera capa MLP: {x_.shape}')
        y = self.act1(self.linear_2(x_))
        #print(f'Salida: {y.shape}')
        return y


def display_image(img, h_res=0.35, v_fov=(-24.9, 2.0), h_fov=(-180, 180), cmap='jet'):
    def scale_xaxis(axis_value, *args):
        if h_fov[0] > 0:
            return int(np.round((axis_value * h_res + h_fov[0])))
        else:
            return int(np.round((axis_value * h_res - abs(h_fov[0]))))

    plt.subplots(1,1, figsize = (30,20))
    plt.imshow(img, cmap=cmap)
    plt.xticks(np.arange(0,len(img[1]),len(img[1])/8))
    formatter = FuncFormatter(scale_xaxis)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel("Ángulo de rotación [º]")
    plt.ylabel("Canales")
    plt.show()

    print(f"Size: {img.shape}")


model_distance = Approx_net(input_channels=1, input_features=6, conv_filters_1=8, conv_filters_2=2, linear_features=12)
model_distance.load_state_dict(torch.load(r'.\models_result\model_cnn1d_min_82-12_Chamfer_sinnorm_no_ego_filtered.pth'))

model_intensity = Approx_net(input_channels=1, input_features=6, conv_filters_1=8, conv_filters_2=2, linear_features=12)
model_intensity.load_state_dict(torch.load(r'.\models_result\model_cnn1d_min_82-12_Chamfer_sinnorm_no_ego_filtered_intensity.pth'))

device = 'cpu'

i = '000793'
path = rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\testing\velodyne\{i}.bin'
pointcloud = read_bin(path)

hrimg_distance = pointcloud_to_range_image(pointcloud, filter_ego_compensed=False, size=(64, 2048), kind='distance') #/ kitti_max_distance
hrimg_intensity = pointcloud_to_range_image(pointcloud, filter_ego_compensed=False, size=(64, 2048), kind='intensity')

indexes = range(0, 64, 2)
lrimg_distance = hrimg_distance[indexes]
lrimg_intensity = hrimg_intensity[indexes]

lrimg_distance = torch.from_numpy(lrimg_distance).view(1,1,lrimg_distance.shape[0],lrimg_distance.shape[1]).to(device)
lrimg_intensity = torch.from_numpy(lrimg_intensity).view(1,1,lrimg_intensity.shape[0],lrimg_intensity.shape[1]).to(device)

hrimg_distance = torch.from_numpy(hrimg_distance).view(1,1,hrimg_distance.shape[0],hrimg_distance.shape[1]).to(device)
hrimg_intensity = torch.from_numpy(hrimg_intensity).view(1,1,hrimg_intensity.shape[0],hrimg_intensity.shape[1]).to(device)


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

output_distance = toutputs_distance.detach().numpy()[0][0]
output_intensity = toutputs_intensity.detach().numpy()[0][0]

#display_image(hrimg_distance)
#display_image(lrimg_distance.detach().cpu().numpy()[0,0])
#display_image(output_distance)

#np.save(r'.\range_images\original.npy', hrimg_distance.detach().cpu().numpy()[0,0])
#np.save(r'.\range_images\original_32.npy', lrimg_distance.detach().cpu().numpy()[0,0])
#np.save(r'.\range_images\interpolada_mired.npy', output_distance)

#pointcloud = range_image_to_pointcloud_with_instensity(output_distance, output_intensity)
#save_PATH = r'.\result_pointclouds\mi_interpolacion.ply'
#save_ply(pointcloud, save_PATH)

pointcloud = range_image_to_pointcloud_with_instensity(output_distance, output_intensity)
save_PATH = rf'.\result_pointclouds\interpolada_testing_{i}.ply'
save_ply(pointcloud, save_PATH)
