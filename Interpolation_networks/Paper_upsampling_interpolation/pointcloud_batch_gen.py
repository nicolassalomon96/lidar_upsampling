#Script para generar dataset con nubes de puntos interpoladas con técnica del paper
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys

from data_gen import *
from pointcloud_utils_functions_v2 import *
device = "cuda" if torch.cuda.is_available() else "cpu"
#device = 'cpu'

pointcloud_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\velodyne'

class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()


class train_data(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = os.listdir(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        indexes = range(0, 64, 2)
        pointcloud_path = os.path.join(self.data_dir, self.data[idx])
        pointcloud = read_bin(pointcloud_path)        
        range_image_4ch = pointcloud_to_range_image(pointcloud, size=(64,2048), filter_ego_compensed=False, kind='all')
        range_image_4ch_low = range_image_4ch[indexes]
        return range_image_4ch, range_image_4ch_low


class Upsampling_Conv(nn.Module):
    def __init__(self, upsampling_factor=2, n_channels=1, padding=(0,1), stride=(1,1)): 
        #padding:(agregar n filas, agregar n columnas)
        #stride(de a cuantas filas me muevo:de a cuantas columnas me muevo)
        super(Upsampling_Conv, self).__init__()

        self.kernel_size = (2,3)
        self.padding = padding
        self.stride = stride
        self.n_channels = n_channels
        self.upsamplig_factor = upsampling_factor
        self.distance_kernel_x2 = torch.tensor([torch.sqrt(torch.tensor(2)), 1, torch.sqrt(torch.tensor(2)), torch.sqrt(torch.tensor(2)), 1, torch.sqrt(torch.tensor(2))], device=device) #Distancias Euclídeas para un kernel de 3x3 desde un pixel central de posición (1,1)
                                                                                                                           #sin considerar los píxeles (1,0) y (1,2) porque esos son los nuevos píxeles a calcular también
        self.distance_kernel_up = torch.tensor([torch.sqrt(torch.tensor(2)), 1, torch.sqrt(torch.tensor(2)), torch.sqrt(torch.tensor(10)), 3, torch.sqrt(torch.tensor(10))], device=device)
        self.distance_kernel_mid = torch.tensor([torch.sqrt(torch.tensor(5)), 2, torch.sqrt(torch.tensor(5)), torch.sqrt(torch.tensor(5)), 2, torch.sqrt(torch.tensor(5))], device=device) 
        self.distance_kernel_down = torch.tensor([torch.sqrt(torch.tensor(10)), 3, torch.sqrt(torch.tensor(10)), torch.sqrt(torch.tensor(2)), 1, torch.sqrt(torch.tensor(2))], device=device)  
        self.lambda_wi = torch.tensor([0.5], device=device)
   
    def forward(self, x):
        n_batches = x.shape[0]
        height = x.shape[2] * self.upsamplig_factor
        width = x.shape[3]
        result = torch.zeros([n_batches, self.n_channels, height, width], dtype=torch.float32, device=device)

        if self.upsamplig_factor == 2:
            even_raw_id = torch.arange(0, result.shape[2], 2)
            odd_raw_id = torch.arange(1, result.shape[2]-1, 2) #El -1 soluciona el problema que causa la última fila de la imagen resultante, al no poser ser escaneada por la convolución

            pixels = self.get_NewPixelsValues(x) #pixels = [batch, 1, filas, columnas]

            result[:,:,even_raw_id,:] = x
            result[:,:,odd_raw_id,:] = pixels
            result = torch.dstack((result[:,:,:-1,:], x[:,:,-1:,:])) #Repito la última fila
            #print(result.shape)
            #result[:,:,-1,:] = x[:,:,-1,:]       

        elif self.upsamplig_factor == 4:
            x_raw_id = torch.arange(0, result.shape[2], 4) #Filas en matriz result ocupadas por valores originales de x
            #new_raw_id = torch.arange(0,result.shape[2],1)[torch.arange(0,result.shape[2],1) % 4 != 0] #Filas en matriz result ocupadas por valores interpolados
            up_raw_id = torch.arange(1,result.shape[2]-3,4) #Filas en matriz result ocupadas por valores interpolados de pixeles superiores, considerando que para interpolar x4 se necesitan 3 nuevos valores
            mid_raw_id = torch.arange(2,result.shape[2]-3,4) #Filas en matriz result ocupadas por valores interpolados de pixeles intermedios, considerando que para interpolar x4 se necesitan 3 nuevos valores
            down_raw_id = torch.arange(3,result.shape[2]-3,4) #Filas en matriz result ocupadas por valores interpolados de pixeles inferiores, considerando que para interpolar x4 se necesitan 3 nuevos valores
            
            pixels = self.get_NewPixelsValues(x)

            result[:,:,x_raw_id,:] = x
            #result[:,:,-3:,:] = x[:,:,-1,:] #Últimas 3 filas iguales
            result = torch.dstack((result[:,:,:-3,:], x[:,:,-1:,:], x[:,:,-1:,:], x[:,:,-1:,:])) #Últimas 3 filas iguales
            result[:,:,up_raw_id,:] = pixels[0]
            result[:,:,mid_raw_id,:] = pixels[1]
            result[:,:,down_raw_id,:] = pixels[2]
            
        else:
            print("ERROR: Wrong Upsampling factor")
        return result  
    
    def weight_function(self, kernel, distance_kernel):
        nonzero_pos = [kernel != 0][0].type(torch.int)
        #print(self.lambda_wi.device, self.distance_kernel.device, kernel.device, nonzero_pos.device)
        y = torch.exp(-self.lambda_wi*distance_kernel) * (2 / (1 + torch.exp(kernel - torch.min(kernel)))) * nonzero_pos #Solo considero los valores que cumplen la condición de que sean mayores que 0
        return y

    def get_NewPixelsValues(self, x):
        windows = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        windows = windows.transpose(1, 2) #Obtener los valores de la ventana o kernel ordenados por fila, donde cada fila representa una ventana serializada

        if self.upsamplig_factor == 2:
            #print(self.distance_kernel_up.shape, windows.shape)
            wi = self.weight_function(windows, self.distance_kernel_x2)
            pixels_value_num = torch.multiply(windows, wi).sum(2) #Sumar todos los elementos de una fila para obtener el valor final del pixel
            pixels_value_den = torch.sum(wi, dim=2)
            pixels_value_den[pixels_value_den == 0] = 1 #Control para evitar errores en la división
            pixels = pixels_value_num/pixels_value_den
            pixels = pixels.view(x.shape[0], x.shape[1], -1, x.shape[-1])

        elif self.upsamplig_factor == 4:
            #print(self.distance_kernel_up.shape, windows.shape)
            wi_up = self.weight_function(windows, self.distance_kernel_up)
            wi_mid = self.weight_function(windows, self.distance_kernel_mid)
            wi_down = self.weight_function(windows, self.distance_kernel_down)
            
            pixels_value_num_up = torch.multiply(windows, wi_up).sum(2) #Sumar todos los elementos de una fila para obtener el valor final del pixel
            pixels_value_num_mid = torch.multiply(windows, wi_mid).sum(2)
            pixels_value_num_down = torch.multiply(windows, wi_down).sum(2)

            pixels_value_den_up = torch.sum(wi_up, dim=2)
            pixels_value_den_mid = torch.sum(wi_mid, dim=2)
            pixels_value_den_down = torch.sum(wi_down, dim=2)

            pixels_value_den_up[pixels_value_den_up == 0] = 1 #Control para evitar errores en la división
            pixels_value_den_mid[pixels_value_den_mid == 0] = 1 #Control para evitar errores en la división
            pixels_value_den_down[pixels_value_den_down == 0] = 1 #Control para evitar errores en la división

            pixels_up = (pixels_value_num_up/pixels_value_den_up).view(x.shape[0], x.shape[1], -1, x.shape[-1])
            pixels_mid = (pixels_value_num_mid/pixels_value_den_mid).view(x.shape[0], x.shape[1], -1, x.shape[-1])
            pixels_down = (pixels_value_num_down/pixels_value_den_down).view(x.shape[0], x.shape[1], -1, x.shape[-1])
            
            pixels = [pixels_up, pixels_mid, pixels_down]

        else:
            print("ERROR: Wrong Upsampling factor") 
        return pixels

    def string(self):
        return f'Lambda_wi = {self.lambda_wi.item()}'


class myNet(nn.Module):
    def __init__(self): 
        super(myNet, self).__init__()
        self.custom_up_layer_1 = Upsampling_Conv()
        #self.custom_up_layer_2 = Upsampling_Conv()
        
    def forward(self, input):
        output =  self.custom_up_layer_1(input)
        #output = self.custom_up_layer_2(x)
        return output


batch_size = 200
custom_data = train_data(pointcloud_path)
train_dataloader = torch.utils.data.DataLoader(custom_data, batch_size=batch_size, shuffle=False)

#model = myNet()
model = Upsampling_Conv(upsampling_factor=2)
model.to(device)

def generate_images(save_path):

    name_idx = 0
    for _, data in enumerate(train_dataloader):
        range_image_4ch, range_image_4ch_low = data       
        range_image_4ch_low = range_image_4ch_low.to(device)

        # Make predictions for this batch
        outputs_intensity = model(range_image_4ch_low[:,:,:,1].unsqueeze(dim=1))
        outputs_x = model(range_image_4ch_low[:,:,:,2].unsqueeze(dim=1))
        outputs_y = model(range_image_4ch_low[:,:,:,3].unsqueeze(dim=1))
        outputs_z = model(range_image_4ch_low[:,:,:,4].unsqueeze(dim=1))

        for i in range(len(outputs_intensity)):
            intensities_serial = outputs_intensity[i,...].cpu().numpy().reshape(-1,1)
            outputs_x_serial = outputs_x[i,...].cpu().numpy().reshape(-1,1)
            outputs_y_serial = outputs_y[i,...].cpu().numpy().reshape(-1,1)
            outputs_z_serial = outputs_z[i,...].cpu().numpy().reshape(-1,1)
            pointcloud = np.hstack([outputs_x_serial, outputs_y_serial, outputs_z_serial, intensities_serial])
            save_name = str(name_idx).zfill(6) + '.bin'
            save_bin(pointcloud, os.path.join(save_path, save_name))
            name_idx += 1


save_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\PointPillars\dataset\paper_upsampling_implementation\training\velodyne'
if not os.path.exists(save_path):
    os.mkdir(save_path)

generate_images(save_path)
