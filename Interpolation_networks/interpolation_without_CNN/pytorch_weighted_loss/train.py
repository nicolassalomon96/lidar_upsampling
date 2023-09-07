import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import kaolin
from data_gen_distance import *
from pointcloud_utils_functions_v2 import *
from object_filtering_pytorch import pointcloud_filter

##################################### VARIABLES #####################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 500
lr = 1e-1
std_percent = 1e-2
dropout_rate = 0.5
l2_weight = 0.01

epoch_number = 0
EPOCHS = 500
best_vloss = 1_000_000

save_folder = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\lidar_upsampling\Interpolation_networks\interpolation_without_CNN\pytorch_weighted_loss\trained_models'
writer_path = os.path.join(save_folder, r'console_outputs.txt')

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

mlp_net = Approx_net(input_size=8, hidden_neurons_1=16, hidden_neurons_2=8, hidden_neurons_3=4, output_size=1)
mlp_net = mlp_net.to(device)

######################################### LOSS ######################################

class ChamferLoss(nn.Module):
    def __init__(self, device):
        super(ChamferLoss, self).__init__()
        self.device = device

    def forward(self, image_pred, image_gt, label_path):
        """
        Compute the Chamfer distance between predicted and ground truth point clouds.

        Args:
            point_cloud_pred (torch.Tensor): Predicted range image tensor of shape [batch_size, num_points, num_dims]
            point_cloud_gt (torch.Tensor): Ground truth range image tensor of shape [batch_size, num_points, num_dims]
            label_path: path to label files [batch_size]
        Returns:
            torch.Tensor: Chamfer distance loss.
        """
        pointcloud_pred = range_image_to_pointcloud_pytorch(image_pred * kitti_max_distance, device) #[batch, num_points, 4]
        pointcloud_gt = range_image_to_pointcloud_pytorch(image_gt * kitti_max_distance, device) #[batch, num_points, 4]

        pointcloud_gt_non_filtered, pointcloud_gt_filtered = pointcloud_filter(pointcloud_gt, label_path) #Listas de tamaño [batch_size] con las nubes de puntos
        pointcloud_pred_non_filtered, pointcloud_pred_filtered = pointcloud_filter(pointcloud_pred, label_path)

        chamfer_loss_filtered = []
        chamfer_loss_non_filtered = []

        for i in range(len(label_path)):
            # Pointclouds must have dimensions [batch, num_points, 3]
            chamfer_distance_filtered = kaolin.metrics.pointcloud.chamfer_distance(pointcloud_pred_filtered[i][:,:3].unsqueeze(0).to(self.device), pointcloud_gt_filtered[i][:,:3].unsqueeze(0).to(self.device))
            chamfer_distance_non_filtered = kaolin.metrics.pointcloud.chamfer_distance(pointcloud_pred_non_filtered[i][:,:3].unsqueeze(0).to(self.device), pointcloud_gt_non_filtered[i][:,:3].unsqueeze(0).to(self.device))
            #print(chamfer_distance_filtered,chamfer_distance_non_filtered)

            chamfer_loss_filtered.append(chamfer_distance_filtered)
            chamfer_loss_non_filtered.append(chamfer_distance_non_filtered)
    
        chamfer_loss_filtered = torch.tensor(chamfer_loss_filtered)
        chamfer_loss_non_filtered = torch.tensor(chamfer_loss_non_filtered)

        chamfer_loss_mean = torch.mean(chamfer_loss_filtered) * 0.7 + torch.mean(chamfer_loss_non_filtered) * 0.3
        return chamfer_loss_mean

loss_fn = ChamferLoss(device=device)
#loss_fn = nn.L1Loss()

############################## OPTIMIZER - LR SCHEDULER ###############################
optimizer = torch.optim.Rprop(mlp_net.parameters(), lr=lr, etas=(0.5,1.5))

################################# WRITER ############################################
def parameters_writer(parameters_list, save_path):
    with open(save_path, 'a+') as f:
        for param in parameters_list:
            f.write(f'{param}, ')
        f.write('\n')

############################## TRAIN LOOP #############################################
hr = torch.ones((1, 1, high_res_height, high_res_width))
row_pos, column_pos = torch.where(hr[0,0] >= 0.0)
odd_row_pos = row_pos[row_pos % 2 != 0] / row_pos.max()
column_pos = column_pos[:column_pos.shape[0]//2] / column_pos.max()
new_pixel_coords = torch.Tensor(list(zip(odd_row_pos, column_pos))[:-1024]) #[:-1024]

def get_windows(x, new_pixel_coords, upscaling_factor=2):
    #def __init__(self, upsampling_factor=2, n_channels=1, padding=(0,1), stride=(1,1)): 
    #padding:(agregar n filas, agregar n columnas)
    #stride(de a cuantas filas me muevo:de a cuantas columnas me muevo)
    kernel_size = (2,3)#(upsampling_factor, upsampling_factor+1) #upsampling x2 --> kernel: 2x3 // upsampling x4 --> kernel:5x5
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


def train_one_epoch(epoch_index, new_pixel_coords):
    running_loss = 0
 
    for j, data in enumerate(train_dataloader):
        #inicio = time.time()
        #print(f'Batch: {j} / {np.floor(len(train_urls)/batch_size)}')
        # Every data instance is an input + label pair
        lrimgs, hrimgs, labels_path = data

        lrimgs = lrimgs.to(device)
        hrimgs = hrimgs.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        
        windows = get_windows(lrimgs, new_pixel_coords)
        #print(f'Windows shape: {windows.shape}')

        pixels = torch.zeros((windows.shape[0], windows.shape[1], 1), device=device)
        for i in range(windows.shape[1]):
            pixels[:,i] = mlp_net(windows[:,i,:])

        pixels = torch.flatten(pixels)
        pixels = pixels.view(lrimgs.shape[0], lrimgs.shape[1], -1, lrimgs.shape[-1])  
        
        pred_image = torch.clone(hrimgs)
        pred_image[:,:,1:pred_image.shape[2]-1:2, :pred_image.shape[3]] = pixels

        # Compute the loss and its gradients
        loss = loss_fn(pred_image, hrimgs, labels_path)
        print(loss)

        #loss = torch.autograd.Variable(loss, requires_grad = True)
        loss.requires_grad = True
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()
   
        # Gather data and report
        running_loss += loss.item()
        #print(f'Tiempo 1 batch = {time.time() - inicio}')
        
    return running_loss / (j + 1)

for epoch in range(EPOCHS):
    inicio = time.time()
   
    #net_parameters = [] #Lista para escribir un .txt con los valores de los parámetros de cada epoca de entrenamiento
    train_parameters = [] #Lista para escribir un .txt con los valores de loss de cada epoca de entrenamiento

    #net_parameters.append(torch.max(mlp_net.linear_one.weight).item())
    #net_parameters.append(torch.max(mlp_net.linear_one.bias).item())
    #net_parameters.append(torch.max(mlp_net.linear_two.weight).item())
    #net_parameters.append(torch.max(mlp_net.linear_two.bias).item())
    #net_parameters.append(torch.max(mlp_net.linear_three.weight).item())
    #net_parameters.append(torch.max(mlp_net.linear_three.bias).item())
    
    #parameters_writer(net_parameters, writer_path)
    
    # Make sure gradient tracking is on, and do a pass over the data
    #for name, param in mlp_net.named_parameters():
    #    print (name, torch.mean(param.data))
    mlp_net.train()
    avg_loss = train_one_epoch(epoch, new_pixel_coords)
    
    # We don't need gradients on to do reporting
    running_vloss = 0.0
    mlp_net.eval()

    with torch.no_grad():
        for k, vdata in enumerate(valid_dataloader):
            vlrimgs, vhrimgs, vlabels_path = vdata
            vlrimgs = vlrimgs.to(device)
            vhrimgs = vhrimgs.to(device)

            vwindows = get_windows(vlrimgs, new_pixel_coords)
            
            vpixels = torch.zeros((vwindows.shape[0], vwindows.shape[1], 1), device=device)
            for i in range(vwindows.shape[1]):
                vpixels[:,i] = mlp_net(vwindows[:,i,:])
            
            vpixels = torch.flatten(vpixels)
            vpixels = vpixels.view(vlrimgs.shape[0], vlrimgs.shape[1], -1, vlrimgs.shape[-1])

            pred_vimage = torch.clone(vhrimgs)
            pred_vimage[:,:,1:pred_vimage.shape[2]-1:2, :pred_vimage.shape[3]] = vpixels
          
            vloss = loss_fn(pred_vimage, vhrimgs, vlabels_path)
            running_vloss += vloss.item()
            
        avg_vloss = running_vloss / (k + 1)

    fin = time.time()
    print(f'Epoch {epoch_number + 1} - Train_loss: {avg_loss} - Valid_loss: {avg_vloss} - lr: {optimizer.param_groups[0]["lr"]} - Tiempo: {(fin-inicio)/60.0} minutos - Estimado: {((EPOCHS-epoch_number-1)*(fin-inicio))/60.0} minutos')
    
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = os.path.join(save_folder, rf'model_816841_kitti3d_Chamfer_Rprop_ep{epoch_number+1}.pth')
        torch.save(mlp_net.state_dict(), model_path)

    train_parameters.append(f'Epoch {epoch_number + 1} - Train_loss: {avg_loss} - Valid_loss: {avg_vloss} - lr: {optimizer.param_groups[0]["lr"]} - Tiempo: {(fin-inicio)/60.0} minutos')
    parameters_writer(train_parameters, writer_path)

    epoch_number += 1

model_path = os.path.join(save_folder, rf'model_816841_kitti3d_Chamfer_Rprop_ep{epoch_number}.pth')
torch.save(mlp_net.state_dict(), model_path)