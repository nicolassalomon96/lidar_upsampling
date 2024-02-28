import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import kaolin
device = "cuda" if torch.cuda.is_available() else "cpu"
#tensorboard --logdir=runs
import time

sys.path.append('..')
from pointcloud_utils_functions_v2 import *
from data_gen_intensity import *

##################################### VARIABLES #####################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 100
lr = 1e-2

epoch_number = 0
EPOCHS = 1000
best_vloss = 1_000_000

writer_path = r'..\models_result\model_cnn1d_min_82-12_Chamfer_xxxxxx_sinnorm_no_ego_filtered_intensity.txt'
#writer_path = r'.\models_result\model_cnn1d_Chamfer_240ep_sinnorm.txt'
#pretrained_model_path = r'.\models_result\mejores\model_cnn1d_mse_82-12_sinnorm.pth'
#pretrained_model_path = r'..\models_result\model_cnn1d_mae_82-12_sinnorm_paper.pth'
#writer = SummaryWriter()

##################################### DATASET - DATALOADER ##########################################

class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()

train_dataset = IterDataset(train_data_generator)
valid_dataset = IterDataset(valid_data_generator)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

##################################### NETWORK #######################################################
'''
class Approx_net(nn.Module):    
    def __init__(self, input_size, hidden_neurons_1, hidden_neurons_2, hidden_neurons_3, output_size):
        super(Approx_net, self).__init__()
        # hidden layer 
        self.linear_one = nn.Linear(input_size, hidden_neurons_1, bias=True)
        self.linear_two = nn.Linear(hidden_neurons_1, hidden_neurons_2, bias=True)
        self.linear_three = nn.Linear(hidden_neurons_2, hidden_neurons_3, bias=True)
        self.linear_four = nn.Linear(hidden_neurons_3, output_size, bias=True)

        #Inicializo los pesos con función He Normal: leí que es mejor si uso ReLU y Leaky_ReLU
        nn.init.kaiming_normal_(self.linear_one.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear_two.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear_three.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear_four.weight, a=0.01, nonlinearity='relu')
        
        self.act = nn.LeakyReLU()
        self.act1 = nn.ReLU()
    
    # prediction function
    def forward(self, x):
        x_ = self.act(self.linear_one(x))
        x_ = self.act(self.linear_two(x_))
        x_ = self.act(self.linear_three(x_))
        y = self.act1(self.linear_four(x_))
        
        return y
'''
class Approx_net(nn.Module):    
    def __init__(self, input_channels, input_features, conv_filters_1, conv_filters_2, linear_features):
        super(Approx_net, self).__init__()
        #cnn layer
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=conv_filters_1, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=conv_filters_1, out_channels=conv_filters_2, kernel_size=1)
        # hidden layer
        self.linear_1 = nn.Linear(in_features=conv_filters_2*input_features+1, out_features=linear_features) #Para considerar el mínimo, se suma 1 a la entrada
        #self.linear_1 = nn.Linear(in_features=conv_filters_2*input_features, out_features=linear_features)
        self.linear_2 = nn.Linear(in_features=linear_features, out_features=1)
        
        self.act = nn.LeakyReLU()
        self.act1 = nn.ReLU()
    
    # prediction function
    def forward(self, x):
        min_value, _ = x.min(dim=1)
        #print(f'Entada: {x.shape}')
        x = x.unsqueeze(1)
        x_ = self.conv1(x)
        #print(f'Salida con1d-1: {x_.shape}')
        x_ = self.conv2(x_)
        #print(f'Salida con1d-2: {x_.shape}')
        x_ = x_.view(x.size(0), -1)
        x_ = torch.hstack((x_, min_value.unsqueeze(1)))
        #print(f'Entrada MLP: {x_.shape}')
        x_ = self.act(self.linear_1(x_))
        #print(f'Salida primera capa MLP: {x_.shape}')
        y = self.act1(self.linear_2(x_))
        #print(f'Salida: {y.shape}')
        return y

mlp_net = Approx_net(input_channels=1, input_features=6, conv_filters_1=8, conv_filters_2=2, linear_features=12)

#mlp_net.load_state_dict(torch.load(pretrained_model_path))
#mlp_net = Approx_net(input_size=8, hidden_neurons_1=16, hidden_neurons_2=8, hidden_neurons_3=4, output_size=1)
mlp_net = mlp_net.to(device)

######################################### LOSS ######################################
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
        #pointcloud_pred = range_image_to_pointcloud_pytorch(image_pred * kitti_max_distance, device)
        #pointcloud_gt = range_image_to_pointcloud_pytorch(image_gt * kitti_max_distance, device)

        pointcloud_pred = range_image_to_pointcloud_pytorch(image_pred * 1.0, device)
        pointcloud_gt = range_image_to_pointcloud_pytorch(image_gt * 1.0, device)

        chamfer_loss = kaolin.metrics.pointcloud.chamfer_distance(pointcloud_pred[:,:,:3].to(self.device), pointcloud_gt[:,:,:3].to(self.device))
        chamfer_loss_mean = torch.mean(chamfer_loss)

        return chamfer_loss_mean

class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        loss = torch.log(torch.cosh(error))
        return torch.mean(loss)

class TukeyLoss(nn.Module):
    def __init__(self, kappa=1.0):
        super(TukeyLoss, self).__init__()
        self.kappa = kappa

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        quadratic_part = torch.where(torch.abs(error) < self.kappa, 0.5 * error**2, self.kappa * (torch.abs(error) - 0.5 * self.kappa))
        return torch.mean(quadratic_part)

#loss_fn = ChamferLoss(device=device)
loss_fn = nn.L1Loss()
#loss_fn = TukeyLoss()
#loss_fn = nn.SmoothL1Loss()
#loss_fn = nn.MSELoss()
#loss_fn = nn.HuberLoss()

############################## OPTIMIZER - LR SCHEDULER ###############################

#optimizer = torch.optim.AdamW(mlp_net.parameters(), lr=lr, weight_decay=0.01)
#optimizer = torch.optim.Rprop(mlp_net.parameters(), lr=lr, etas=(0.5,1.5))
optimizer = torch.optim.Adam(mlp_net.parameters(), lr=lr, weight_decay=1e-5)

#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, last_epoch=400, gamma=0.5) #Cada x épocas lr = lr * gamma
#lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=5, last_epoch=30)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=2, eta_min=lr/100)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=10)
 
################################# WRITER ############################################
def parameters_writer(parameters_list, save_path):
    with open(save_path, 'a+') as f:
        for param in parameters_list:
            f.write(f'{param}, ')
        f.write('\n')

############################## TRAIN LOOP #############################################
'''
hr = torch.ones((1, 1, high_res_height, high_res_width))
#hr = torch.ones((1, 1, high_res_height, 2048))
row_pos, column_pos = torch.where(hr[0,0] >= 0.0)
odd_row_pos = row_pos[row_pos % 2 != 0] / row_pos.max()
column_pos = column_pos[:column_pos.shape[0]//2] / column_pos.max()
new_pixel_coords = torch.Tensor(list(zip(odd_row_pos, column_pos))[:-2048]) #[:-1024]

def get_windows(x, new_pixel_coords, upscaling_factor=2):
    kernel_size = (2,3)
    padding = (0,1)
    stride = (1,1)
    new_pixel_coords_batch = new_pixel_coords.unsqueeze(0).repeat(x.shape[0],1,1).to(device)

    if upscaling_factor == 2:
        windows = F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)
        windows = windows.transpose(1, 2) #Obtener los valores de la ventana o kernel ordenados por fila, donde cada fila representa una ventana serializada
        windows = torch.dstack((windows, new_pixel_coords_batch))
    else:
        print("ERROR: Wrong Upsampling factor")
    return windows  

'''
def get_windows(x, upscaling_factor=2):
        kernel_size = (2,3)
        padding = (0,1)
        stride = (1,1)
        
        if upscaling_factor == 2:
            windows = F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)
            windows = windows.transpose(1, 2) #Obtener los valores de la ventana o kernel ordenados por fila, donde cada fila representa una ventana serializada
            windows = windows.to(device)
            #windows_min, _ = windows.min(dim=2)
            #windows = torch.dstack((windows, windows_min))
        else:
            print("ERROR: Wrong Upsampling factor")
        return windows

def train_one_epoch(epoch_index):
    running_loss = 0
 
    for j, data in enumerate(train_dataloader):
        #inicio = time.time()
        lrimg_intensity, hrimg_intensity  = data
        lrimg_intensity = lrimg_intensity.to(device)
        hrimg_intensity = hrimg_intensity.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        windows = get_windows(lrimg_intensity)
        #windows = get_windows(lrimg_distance, new_pixel_coords)
        windows_stack = windows.reshape(windows.shape[0] * windows.shape[1], windows.shape[2])
        pixels = mlp_net(windows_stack)#.squeeze()
        pixels = pixels.view(lrimg_intensity.shape[0], lrimg_intensity.shape[1], -1, lrimg_intensity.shape[-1])

        real_pixels = hrimg_intensity[:,:,1:hrimg_intensity.shape[2]-1:2, :hrimg_intensity.shape[3]]

        # Compute the loss and its gradients
        loss = loss_fn(pixels, real_pixels)
        #print(loss.requires_grad)

        # compute penalty only for net.hidden parameters
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        #print(f'Tiempo 1 batch = {time.time() - inicio}')
        
    return running_loss / (j + 1)

for epoch in range(EPOCHS):
    inicio = time.time()
   
    train_parameters = [] #Lista para escribir un .txt con los valores de loss de cada epoca de entrenamiento
    
    # Make sure gradient tracking is on, and do a pass over the data
    #for name, param in mlp_net.named_parameters():
    #    print (name, torch.mean(param.data))
    mlp_net.train()
    avg_loss = train_one_epoch(epoch)

    # Write the average loss to TensorBoard
    #writer.add_scalar('training_loss', avg_loss, epoch)
    
    # We don't need gradients on to do reporting
    running_vloss = 0.0
    mlp_net.eval()

    with torch.no_grad():
        for k, vdata in enumerate(valid_dataloader):
            vlrimg_intensity, vhrimg_intensity  = vdata
            vlrimg_intensity = vlrimg_intensity.to(device)
            vhrimg_intensity = vhrimg_intensity.to(device)

            vwindows = get_windows(vlrimg_intensity)                  
            #vwindows = get_windows(vlrimg_distance, new_pixel_coords)
            vwindows_stack = vwindows.reshape(vwindows.shape[0] * vwindows.shape[1], vwindows.shape[2])
            vpixels = mlp_net(vwindows_stack)#.squeeze()
            vpixels = vpixels.view(vlrimg_intensity.shape[0], vlrimg_intensity.shape[1], -1, vlrimg_intensity.shape[-1])

            real_vpixels = vhrimg_intensity[:,:,1:vhrimg_intensity.shape[2]-1:2, :vhrimg_intensity.shape[3]]
            vloss = loss_fn(vpixels, real_vpixels)
            running_vloss += vloss.item()
            
        avg_vloss = running_vloss / (k + 1)

        lr_scheduler.step(avg_vloss) #Para usar con ReduceLROnPlateau
        #lr_scheduler.step()
    # Write the average loss to TensorBoard
    #writer.add_scalar('valid_loss', avg_vloss, epoch)

    fin = time.time()
    print(f'Epoch {epoch_number + 1} - Train_loss: {avg_loss} - Valid_loss: {avg_vloss} - lr: {optimizer.param_groups[0]["lr"]} - Tiempo: {(fin-inicio)/60.0} minutos - Estimado: {((EPOCHS-epoch_number-1)*(fin-inicio))/60.0} minutos')
    
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = rf'..\models_result\model_cnn1d_min_82-12_Chamfer_xxxxxx_sinnorm_no_ego_filtered_intensity_1024.pth'
        torch.save(mlp_net.state_dict(), model_path)

    train_parameters.append(f'Epoch {epoch_number + 1} - Train_loss: {avg_loss} - Valid_loss: {avg_vloss} - lr: {optimizer.param_groups[0]["lr"]}')
    parameters_writer(train_parameters, writer_path)

    epoch_number += 1

#writer.close()
