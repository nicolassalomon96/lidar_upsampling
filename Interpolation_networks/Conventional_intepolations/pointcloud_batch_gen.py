#Script para generar dataset con nubes de puntos interpoladas según técnica propia
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pointcloud_utils_functions_v2 import *
from data_gen_all import *
device = "cuda" if torch.cuda.is_available() else "cpu"

pointcloud_saved_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\lidar_upsampling\Classification_networks\PointPillars\dataset\kitti_bilineal\training\velodyne'

############################################## RED DE INTERPOLACIÓN PESADA BASADA EN PAPER ####################################################
class Upsampling_Conv(nn.Module):
    def __init__(self, upsampling_factor=2, n_channels=1, kind='bilinear'):
        super(Upsampling_Conv, self).__init__()    
        self.upsamplig_factor = upsampling_factor
        self.kind = kind
        self.n_channels = n_channels
           
    def forward(self, x):
        height = x.shape[2] * self.upsamplig_factor
        width = x.shape[3]
        result = F.interpolate(x, size=(height, width), mode=self.kind)       
        return result  

class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()
    
train_dataset = IterDataset(data_generator)

batch_size = 1000
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

################################################ CREACIÓN DEL MODELO Y PROCESADO DE DATOS ####################################
model = Upsampling_Conv(kind='bilinear')
pointcloud_number = 0
for i, data in enumerate(tqdm(dataloader)):
    lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity = data
    lrimg_distance = lrimg_distance.to(device)
    lrimg_intensity = lrimg_intensity.to(device)

    ###################################### PREDICCIÓN DE PIXELES DE DISTANCIA #########################################
    outputs_distance = model(lrimg_distance)
    pointcloud_gen_batch_distance = range_image_to_pointcloud_pytorch(outputs_distance, device)
    ###################################### PREDICCIÓN DE PIXELES DE INTENSIDAD #########################################
    outputs_intensity = model(lrimg_intensity)

    for j, pointcloud in enumerate(pointcloud_gen_batch_distance):
        pointcloud_join = torch.hstack([pointcloud[:,:3], outputs_intensity[j,0,:,:].reshape(-1,1)])
        save_path = os.path.join(pointcloud_saved_path, f'{str(pointcloud_number).zfill(6)}.bin')
        save_bin(pointcloud_join.cpu().detach().numpy(), save_path)
        pointcloud_number += 1