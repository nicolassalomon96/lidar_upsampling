import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\otras_arquitecturas\3_pytorch_interpolation')
from data_gen_distance import *
from pointcloud_utils_functions_v2 import *

class Approx_net(nn.Module):    
    def __init__(self, input_size, hidden_neurons_1, hidden_neurons_2, hidden_neurons_3, output_size):
        super(Approx_net, self).__init__()
        # hidden layer 
        self.linear_one = nn.Linear(input_size, hidden_neurons_1, bias=True)
        self.linear_two = nn.Linear(hidden_neurons_1, hidden_neurons_2, bias=True)
        self.linear_three = nn.Linear(hidden_neurons_2, output_size, bias=True)
        #self.linear_three = nn.Linear(hidden_neurons_2, hidden_neurons_3, bias=True)
        #self.linear_four = nn.Linear(hidden_neurons_3, output_size, bias=True)

        self.act = nn.LeakyReLU()
        self.act1 = nn.ReLU()
    
    # prediction function
    def forward(self, x):
        x_ = self.act(self.linear_one(x))
        x_ = self.act(self.linear_two(x_))
        #x_ = self.act(self.linear_three(x_))
        #y = self.act1(self.linear_four(x_))
        y = self.act1(self.linear_three(x_))
        return y

#mlp_net_distance = Approx_net(input_size=8, hidden_neurons_1=20, hidden_neurons_2=10, hidden_neurons_3=6, output_size=1)
mlp_net_distance = Approx_net(input_size=8, hidden_neurons_1=16, hidden_neurons_2=8, hidden_neurons_3=6, output_size=1)
#mlp_net_distance.load_state_dict(torch.load(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\otras_arquitecturas\3_pytorch_interpolation\results\mlp_8201061\model_Leaky_ReLU_8201061_30ep_fulldataset_Chamfer\model_Leaky_ReLU_8201061_reduced_Chamfer_LBFGS_ep49_val025_bz512.pth'))
mlp_net_distance.load_state_dict(torch.load(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\otras_arquitecturas\3_pytorch_interpolation\results\mlp_8201061\model_Leaky_ReLU_8201061_30ep_fulldataset_Chamfer\model_Leaky_ReLU_81681_fulldataset_Chamfer_Rprop_ep81_val046.pth'))

device = 'cpu'

#path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\testing\velodyne\000001.bin'
path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\Kitti-Dataset-master\2011_09_26\2011_09_26_drive_0001_sync\velodyne_points\data\0000000000.bin'
pointcloud = read_bin(path)
#hr_range_distance_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=True, kind='distance', size=(64,1024)) / kitti_max_distance
hr_range_distance_intensity_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=True, kind='both', size=(64,1024))
hr_range_distance_image = hr_range_distance_intensity_image[:,:,0] / kitti_max_distance
hr_range_intensity_image = hr_range_distance_intensity_image[:,:,1]

#print(hr_range_distance_image[:,:,0].shape, hr_range_distance_image[:,:,0].max())
#print(hr_range_distance_image[:,:,1].shape, hr_range_distance_image[:,:,1].max())

downsampling_factor = 2
ind = np.arange(0,64,downsampling_factor)
lr_range_distance_image = hr_range_distance_image[ind]
lr_range_intensity_image = hr_range_intensity_image[ind]

#print(lr_range_distance_image.shape, hr_range_distance_image.shape, hr_range_intensity_image.shape)
#print(lr_range_distance_image.max(), hr_range_distance_image.max(), hr_range_intensity_image.max())
#print(lr_range_distance_image[:,:,0].shape, lr_range_distance_image[:,:,0].max())
#print(lr_range_distance_image[:,:,1].shape, lr_range_distance_image[:,:,1].max())

#input = torch.from_numpy(np.moveaxis(lr_range_distance_image,-1,0)).unsqueeze(0)
#output = torch.from_numpy(np.moveaxis(hr_range_distance_image[:,:,:1],-1,0)).unsqueeze(0)
#output = torch.from_numpy(np.moveaxis(hr_range_distance_image).view(1,2,hr_range_distance_image.shape[0],hr_range_distance_image.shape[1])
input_distance = torch.from_numpy(lr_range_distance_image).view(1,1,lr_range_distance_image.shape[0],lr_range_distance_image.shape[1])
output = torch.from_numpy(hr_range_distance_image).view(1,1,hr_range_distance_image.shape[0],hr_range_distance_image.shape[1])

#print(input.shape, input[:,:1,:,:].max(), input[:,1:,:,:].max())
#display_range_image(input[0,0,:,:].cpu().detach().numpy())
#display_range_image(input[0,1,:,:].cpu().detach().numpy())
#print(output.shape, output.max())

hr = torch.ones((1, 1, high_res_height, high_res_width))
#hr = torch.ones((1, 1, 64, 2048))
row_pos, column_pos = torch.where(hr[0,0] >= 0.0)
odd_row_pos = row_pos[row_pos % 2 != 0] / row_pos.max()
column_pos = column_pos[:column_pos.shape[0]//2] / column_pos.max()
new_pixel_coords = torch.Tensor(list(zip(odd_row_pos, column_pos))[:-1024]) #[:-1024]
new_pixel_coords = new_pixel_coords.unsqueeze(0).to(device)

def get_windows(x, new_pixel_coords, upscaling_factor=2):
    kernel_size = (2,3)
    padding = (0,1)
    stride = (1,1)
    
    if upscaling_factor == 2:
        windows = F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)
        windows = windows.transpose(1, 2) #Obtener los valores de la ventana o kernel ordenados por fila, donde cada fila representa una ventana serializada
        windows = windows.to(device)
        windows = torch.dstack((windows, new_pixel_coords))  
        #windows_distance = F.unfold(x[:,:1,:,:], kernel_size=kernel_size, padding=padding, stride=stride)
        #windows_distance = windows_distance.transpose(1, 2) #Obtener los valores de la ventana o kernel ordenados por fila, donde cada fila representa una ventana serializada
        #windows_intensity = F.unfold(x[:,1:,:,:], kernel_size=kernel_size, padding=padding, stride=stride)
        #windows_intensity = windows_intensity.transpose(1, 2) #Obtener los valores de la ventana o kernel ordenados por fila, donde cada fila representa una ventana serializada
        #windows = torch.dstack((windows_distance, windows_intensity, new_pixel_coords))     
    else:
        print("ERROR: Wrong Upsampling factor")
    return windows


twindows = get_windows(input_distance, new_pixel_coords)
mlp_net_distance = mlp_net_distance.to(device)      

pixels = torch.empty((twindows.shape[0], twindows.shape[1], 1), device=device)
for i in range(twindows.shape[1]):
    pixels[:,i] = mlp_net_distance(twindows[:,i,:])

pixels = torch.flatten(pixels)
pixels = pixels.view(input_distance.shape[0], 1, -1, input_distance.shape[-1])

#print(pixels.shape)

toutputs = torch.zeros_like(output, device=device)
toutputs[:,:,:toutputs.shape[2]:2, :toutputs.shape[3]] = input_distance[:,:1,:,:]
toutputs[:,:,1:toutputs.shape[2]-1:2, :toutputs.shape[3]] = pixels
toutputs = torch.dstack((toutputs[:,:,:-1,:], input_distance[:,:1,-1:,:])) #Repito la última fila   

#print(toutputs.shape)

input_image = input_distance.numpy()[0][0]
output_image = toutputs.detach().numpy()[0][0]

#pointcloud = range_image_to_pointcloud_with_instensity(lr_range_distance_image * kitti_max_distance, lr_range_intensity_image)
pointcloud = range_image_to_pointcloud_with_instensity(output_image * kitti_max_distance, hr_range_intensity_image)
#pointcloud = range_image_to_pointcloud(hr_range_distance_image * kitti_max_distance)
save_PATH = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\pruebas_resultados\b.bin'
#save_ply(pointcloud, save_PATH)
save_bin(pointcloud, save_PATH)

