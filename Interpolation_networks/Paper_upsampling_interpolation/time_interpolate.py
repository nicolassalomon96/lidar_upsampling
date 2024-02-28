#Script para medir el tiempo requerido para interpolar las imágenes de rango empleando técnica del paper
import time
from pointcloud_utils_functions_v2 import *
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pointcloud_utils_functions_v2 import *
device = "cuda" if torch.cuda.is_available() else "cpu"

kitti_pointcloud_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\velodyne\001068.bin'
kitti_image_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training\image_2\001068.png'

pointcloud = read_bin(kitti_pointcloud_path)
range_distance_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=False, size=(64,2048), kind = 'distance')
range_intensity_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=False, size=(64,2048), kind='intensity')

indexes = range(0, 64, 2)
lrimg_distance = torch.from_numpy(range_distance_image[indexes])
lrimg_intensity = torch.from_numpy(range_intensity_image[indexes])

class Upsampling_Conv(nn.Module):
    def __init__(self, upsampling_factor=2, n_channels=1, padding=(0,1), stride=(1,1)): 
        super(Upsampling_Conv, self).__init__()

        self.kernel_size = (2,3)
        self.padding = padding
        self.stride = stride
        self.n_channels = n_channels
        self.upsamplig_factor = upsampling_factor
        self.distance_kernel_x2 = torch.tensor([torch.sqrt(torch.tensor(2)), 1, torch.sqrt(torch.tensor(2)), torch.sqrt(torch.tensor(2)), 1, torch.sqrt(torch.tensor(2))], device=device) #Distancias Euclídeas para un kernel de 3x3 desde un pixel central de posición (1,1)
                                                                                                                           #sin considerar los píxeles (1,0) y (1,2) porque esos son los nuevos píxeles a calcular también
        self.lambda_wi = torch.tensor([0.5], device=device)#nn.Parameter(torch.tensor([0.5]))
   
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
        elif self.upsamplig_factor == 4:
            pass
        else:
            print("ERROR: Wrong Upsampling factor")
        return result  
    
    def weight_function(self, kernel, distance_kernel):
        nonzero_pos = [kernel != 0][0].type(torch.int)
        nonzero_pos = nonzero_pos.to(device)
        y = torch.exp(-self.lambda_wi*distance_kernel) * (2 / (1 + torch.exp(kernel - torch.min(kernel)))) * nonzero_pos #Solo considero los valores que cumplen la condición de que sean mayores que 0
        return y

    def get_NewPixelsValues(self, x):
        windows = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        windows = windows.transpose(1, 2) #Obtener los valores de la ventana o kernel ordenados por fila, donde cada fila representa una ventana serializada
        windows = windows.to(device)

        if self.upsamplig_factor == 2:
            wi = self.weight_function(windows, self.distance_kernel_x2)
            pixels_value_num = torch.multiply(windows, wi).sum(2) #Sumar todos los elementos de una fila para obtener el valor final del pixel
            pixels_value_den = torch.sum(wi, dim=2)
            pixels_value_den[pixels_value_den == 0] = 1 #Control para evitar errores en la división
            pixels = pixels_value_num/pixels_value_den
            pixels = pixels.view(x.shape[0], x.shape[1], -1, x.shape[-1])

        elif self.upsamplig_factor == 4:
            pass
        else:
            print("ERROR: Wrong Upsampling factor") 
        return pixels

    def string(self):
        return f'Lambda_wi = {self.lambda_wi.item()}'

model = Upsampling_Conv(upsampling_factor=2)
model.to(device)

lrimg_distance = lrimg_distance.unsqueeze(0).unsqueeze(0)
lrimg_intensity = lrimg_intensity.unsqueeze(0).unsqueeze(0)
lrimg_distance = lrimg_distance.to(device)
lrimg_intensity = lrimg_intensity.to(device)

test_iterations = 10
total_time = []
for i in range(test_iterations):
    start = time.time()
    outputs_distance = model(lrimg_distance)
    outputs_intensity = model(lrimg_intensity)
    end = time.time()
    total_time.append(end - start)

print(f'Tiempo de interpolación: {sum(total_time[1:]) / (test_iterations-1)} segundos')