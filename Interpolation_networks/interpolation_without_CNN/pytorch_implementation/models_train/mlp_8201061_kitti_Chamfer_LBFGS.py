import sys
sys.path.append(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\otras_arquitecturas\3_pytorch_interpolation')
sys.path.append(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\main\main_16to64\utils')

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import time
import kaolin
from data_gen_distance import *
device = "cuda" if torch.cuda.is_available() else "cpu"
#torch.set_default_dtype(torch.float64)

from pointcloud_utils_functions_v2 import *

class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()
    
class ChamferLoss(nn.Module):
    def __init__(self, device):
        super(ChamferLoss, self).__init__()
        self.device = device

    def forward(self, image_pred, image_gt):
        """
        Compute the Chamfer distance between predicted and ground truth point clouds.

        Args:
            point_cloud_pred (torch.Tensor): Predicted point cloud tensor of shape (batch_size, num_points, num_dims).
            point_cloud_gt (torch.Tensor): Ground truth point cloud tensor of shape (batch_size, num_points, num_dims).

        Returns:
            torch.Tensor: Chamfer distance loss.
        """
        #mseloss = F.mse_loss(image_pred[:,:,:10,:], image_gt[:,:,:10,:])
        #l1loss = F.l1_loss(image_pred, image_gt)

        pointcloud_pred = range_image_to_pointcloud_pytorch(image_pred * kitti_max_distance, device)
        pointcloud_gt = range_image_to_pointcloud_pytorch(image_gt * kitti_max_distance, device)

        #batch_size, num_points, num_dims = point_cloud_pred.size()
        chamfer_loss = kaolin.metrics.pointcloud.chamfer_distance(pointcloud_pred[:,:,:3].to(self.device), pointcloud_gt[:,:,:3].to(self.device))
        chamfer_loss_mean = torch.mean(chamfer_loss)

        return chamfer_loss_mean #+ l1loss

class Approx_net(nn.Module):    
    def __init__(self, input_size, hidden_neurons_1, hidden_neurons_2, hidden_neurons_3, output_size):
        super(Approx_net, self).__init__()
        # hidden layer 
        self.linear_one = nn.Linear(input_size, hidden_neurons_1, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=hidden_neurons_1)
        self.linear_two = nn.Linear(hidden_neurons_1, hidden_neurons_2, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=hidden_neurons_2)
        #self.linear_three = nn.Linear(hidden_neurons_2, hidden_neurons_3, bias=True)
        self.linear_three = nn.Linear(hidden_neurons_2, output_size, bias=True)
        #self.bn3 = nn.BatchNorm1d(num_features=hidden_neurons_3)
        #self.linear_four = nn.Linear(hidden_neurons_3, output_size, bias=True)
        nn.init.xavier_normal_(self.linear_one.weight)
        nn.init.xavier_normal_(self.linear_two.weight)
        nn.init.xavier_normal_(self.linear_three.weight)
        #nn.init.xavier_normal_(self.linear_four.weight)
        self.act = nn.LeakyReLU()
        #self.act1 = nn.Sigmoid()
        self.act1 = nn.ReLU()
    
    # prediction function
    def forward(self, x):
        #x_ = self.act(self.linear_one(x))
        #x_ = self.act(self.linear_two(x_))
        #x_ = self.act(self.linear_three(x_))
        #y = self.act1(self.linear_four(x_))
        x_ = self.act(self.bn1(self.linear_one(x)))
        x_ = self.act(self.bn2(self.linear_two(x_)))
        #x_ = self.act(self.bn3(self.linear_three(x_)))
        y = self.act1(self.linear_three(x_))
        #y = self.act1(self.linear_four(x_))
        return y

mlp_net = Approx_net(input_size=8, hidden_neurons_1=10, hidden_neurons_2=6, hidden_neurons_3=6, output_size=1)
mlp_net = mlp_net.to(device)

upscaling_factor = 2
batch_size = 512

train_dataset = IterDataset(train_data_generator)
valid_dataset = IterDataset(valid_data_generator)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
#lr=0.1
loss_fn = ChamferLoss(device=device)
#loss_fn = nn.L1Loss()
#optimizer = torch.optim.Adam(mlp_net.parameters(), lr=lr)
#optimizer = torch.optim.Rprop(mlp_net.parameters(), lr=lr)
optimizer = torch.optim.LBFGS(mlp_net.parameters(), lr=1.0, max_iter=20)#, history_size=100)
#lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1,last_epoch=-1) #Cada 10 épocas lr = lr * gamma
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=1, eta_min=0.01)
iters = len(train_urls) 

_, hr = next(iter(train_dataloader))
row_pos,column_pos = torch.where(hr[0,0] >= 0)
odd_row_pos = row_pos[row_pos % 2 != 0] / row_pos.max()
column_pos = column_pos[:column_pos.shape[0]//2] / column_pos.max()
new_pixel_coords = torch.Tensor(list(zip(odd_row_pos, column_pos))[:-1024])

def get_windows(x, new_pixel_coords, upscaling_factor=2):
    #def __init__(self, upsampling_factor=2, n_channels=1, padding=(0,1), stride=(1,1)): 
    #padding:(agregar n filas, agregar n columnas)
    #stride(de a cuantas filas me muevo:de a cuantas columnas me muevo)
    kernel_size = (2,3)#(upsampling_factor, upsampling_factor+1) #upsampling x2 --> kernel: 2x3 // upsampling x4 --> kernel:5x5
    padding = (0,1)
    stride = (1,1)
    n_channels = 1
    n_batches = x.shape[0]
    height = x.shape[2] * upscaling_factor
    width = x.shape[3]
    new_pixel_coords_batch = new_pixel_coords.unsqueeze(0).repeat(x.shape[0],1,1).to(device)
    
    if upscaling_factor == 2:
        windows = F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)
        windows = windows.transpose(1, 2) #Obtener los valores de la ventana o kernel ordenados por fila, donde cada fila representa una ventana serializada
        #print(windows[10,2000])
        #print(new_pixel_coords[10,2000])
        windows = torch.dstack((windows, new_pixel_coords_batch))
        #print(windows[10,2000])       
    else:
        print("ERROR: Wrong Upsampling factor")
    return windows  

def display_image(img, title, figsize=(10,4), cmap='gray'):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.show()

def train_one_epoch(epoch_index, new_pixel_coords):
    running_loss = 0
 
    for j, data in enumerate(train_dataloader):
        #print(f'Batch: {j} / {np.floor(len(train_urls)/batch_size)}')
        # Every data instance is an input + label pair
        lrimgs, hrimgs = data
        lrimgs = lrimgs.to(device)
        hrimgs = hrimgs.to(device)
        '''
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        windows = get_windows(lrimgs, new_pixel_coords)
        #print(f'Windows shape: {windows.shape}')

        pixels = torch.empty((windows.shape[0], windows.shape[1], 1), device=device)
        for i in range(windows.shape[1]):
            pixels[:,i] = mlp_net(windows[:,i,:])

        pixels = torch.flatten(pixels)
        pixels = pixels.view(lrimgs.shape[0], lrimgs.shape[1], -1, lrimgs.shape[-1])  
        
        #odd_raw_id = torch.arange(1, hrimgs.shape[2]-1, 2)
        real_pixels = hrimgs[:,:,1:hrimgs.shape[2]-1:2, :hrimgs.shape[3]]

        # Compute the loss and its gradients
        loss = loss_fn(pixels, real_pixels)
        #print(loss)

        loss.backward()
        
        # Adjust learning weights
        optimizer.step()
        '''
        def closure():
            if torch.is_grad_enabled():
                # Zero your gradients for every batch!
                optimizer.zero_grad()

            windows = get_windows(lrimgs, new_pixel_coords)
            #print(f'Windows shape: {windows.shape}')

            pixels = torch.empty((windows.shape[0], windows.shape[1], 1), device=device)
            for i in range(windows.shape[1]):
                pixels[:,i] = mlp_net(windows[:,i,:])

            pixels = torch.flatten(pixels)
            pixels = pixels.view(lrimgs.shape[0], lrimgs.shape[1], -1, lrimgs.shape[-1])  
            
            real_pixels = hrimgs[:,:,1:hrimgs.shape[2]-1:2, :hrimgs.shape[3]]

            #print(f'pixels: {pixels.max().item(), pixels.min().item(), torch.std_mean(pixels)}')

            # Compute the loss and its gradients
            loss = loss_fn(pixels, real_pixels)
            #print(loss)
            if loss.requires_grad:
                loss.backward()
    
            return loss
            
        # Adjust learning weights
        optimizer.step(closure)

        lr_scheduler.step(epoch_index + j / iters)
        
        # calculate the loss again for monitoring
        windows = get_windows(lrimgs, new_pixel_coords)
        pixels = torch.empty((windows.shape[0], windows.shape[1], 1), device=device)
        for i in range(windows.shape[1]):
            pixels[:,i] = mlp_net(windows[:,i,:])
        pixels = torch.flatten(pixels)
        pixels = pixels.view(lrimgs.shape[0], lrimgs.shape[1], -1, lrimgs.shape[-1])  
        real_pixels = hrimgs[:,:,1:hrimgs.shape[2]-1:2, :hrimgs.shape[3]]

        loss = loss_fn(pixels, real_pixels)
        
        # Gather data and report
        running_loss += loss.item()
        
    return running_loss / (j + 1)

epoch_number = 0

EPOCHS = 100

best_vloss = 1_000_000

for epoch in range(EPOCHS):
    inicio = time.time()
    # Make sure gradient tracking is on, and do a pass over the data
    #for name, param in mlp_net.named_parameters():
    #    print (name, param.data)
    mlp_net.train()
    avg_loss = train_one_epoch(epoch_number, new_pixel_coords)

    # We don't need gradients on to do reporting
    running_vloss = 0.0
    #mlp_net.eval()
    with torch.no_grad():
        for k, vdata in enumerate(valid_dataloader):
            vlrimgs, vhrimgs = vdata
            vlrimgs, vhrimgs = vlrimgs.to(device), vhrimgs.to(device)
            
            vwindows = get_windows(vlrimgs, new_pixel_coords)
            
            vpixels = torch.empty((vwindows.shape[0], vwindows.shape[1], 1), device=device)
            for i in range(vwindows.shape[1]):
                vpixels[:,i] = mlp_net(vwindows[:,i,:])
                
            vpixels = torch.flatten(vpixels)
            vpixels = vpixels.view(vlrimgs.shape[0], vlrimgs.shape[1], -1, vlrimgs.shape[-1])

            real_vpixels = vhrimgs[:,:,1:vhrimgs.shape[2]-1:2, :vhrimgs.shape[3]]
            
            #print(f'vpixels: {vpixels.max().item(), vpixels.min().item(), torch.std_mean(vpixels)}')

            vloss = loss_fn(vpixels, real_vpixels)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / (k + 1)
    fin = time.time()
    #print(f'Epoch {epoch_number + 1} - Train_loss: {avg_loss} - Valid_loss: {avg_vloss} - lr: {optimizer.param_groups[0]["lr"]} - Tiempo: {(fin-inicio)/60.0} minutos - Estimado: {((EPOCHS-epoch_number-1)*(fin-inicio))/60.0} minutos')
    print(f'Epoch {epoch_number + 1} - Train_loss: {avg_loss} - Valid_loss: {avg_vloss} - lr: {optimizer.param_groups[0]["lr"]} - Tiempo: {(fin-inicio)/60.0} minutos - Estimado: {((EPOCHS-epoch_number-1)*(fin-inicio))/60.0} minutos')
    #if ((epoch_number+1) == 20 or (epoch_number+1) == 300):
    #    lr_scheduler.step()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\otras_arquitecturas\3_pytorch_interpolation\results\model_Leaky_ReLU_8201061_30ep_fulldataset_Chamfer\model_Leaky_ReLU_8201061_reduced_Chamfer_LBFGS_lr1_ep{epoch_number+1}.pth'
        torch.save(mlp_net.state_dict(), model_path)
 
    epoch_number += 1

model_path = rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\otras_arquitecturas\3_pytorch_interpolation\results\model_Leaky_ReLU_8201061_30ep_fulldataset_Chamfer\model_Leaky_ReLU_8201061_reduced_Chamfer_LBFGS_lr1_ep{epoch_number}.pth'
torch.save(mlp_net.state_dict(), model_path)