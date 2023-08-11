import sys
sys.path.append(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\otras_arquitecturas\3_pytorch_interpolation')

#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
import time
import kaolin
from data_gen_intensity import *
from pointcloud_utils_functions_v2 import *
import torch_optimizer as optim
#torch.set_default_dtype(torch.float32)

##################################### VARIABLES #####################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 600#512
lr = 1e-1
std_percent = 1e-1
dropout_rate = 0.5
l2_weight = 0.01

epoch_number = 0
EPOCHS = 500
best_vloss = 1_000_000

writer_path = 'parameters_Rprop_intensity.txt'

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
        #self.bn1 = nn.BatchNorm1d(num_features=hidden_neurons_1)
        #self.dropout_1 = nn.Dropout(dropout_rate)
        self.linear_two = nn.Linear(hidden_neurons_1, hidden_neurons_2, bias=True)
        #self.bn2 = nn.BatchNorm1d(num_features=hidden_neurons_2)
        #self.dropout_2 = nn.Dropout(dropout_rate)
        self.linear_three = nn.Linear(hidden_neurons_2, hidden_neurons_3, bias=True)
        #self.linear_three = nn.Linear(hidden_neurons_2, output_size, bias=True)
        #self.bn3 = nn.BatchNorm1d(num_features=hidden_neurons_3)
        #self.dropout_3 = nn.Dropout(dropout_rate)
        self.linear_four = nn.Linear(hidden_neurons_3, output_size, bias=True)
        #self.linear_four = nn.Linear(hidden_neurons_3, hidden_neurons_4, bias=True)
        #self.linear_five = nn.Linear(hidden_neurons_4, output_size, bias=True)

        #Inicializo los pesos con función He Normal: leí que es mejor si uso ReLU y Leaky_ReLU
        nn.init.kaiming_normal_(self.linear_one.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear_two.weight, a=0.01, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.linear_three.weight, a=0.01, nonlinearity='leaky_relu')
        #nn.init.kaiming_normal_(self.linear_three.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear_four.weight, a=0.01, nonlinearity='relu')
        #nn.init.kaiming_normal_(self.linear_five.weight, nonlinearity='relu')

        #nn.init.normal_(self.linear_one.weight, 0, 0.01)
        #nn.init.normal_(self.linear_one.bias, 0, 0.01)
        #nn.init.normal_(self.linear_two.weight, 0, 0.01)
        #nn.init.normal_(self.linear_two.bias, 0, 0.01)
        #nn.init.normal_(self.linear_three.weight, 0, 0.01)
        #nn.init.normal_(self.linear_three.bias, 0, 0.01)
        #nn.init.normal_(self.linear_four.weight, 0, 0.01)
        #nn.init.normal_(self.linear_four.bias, 0, 0.01)
        
        self.act = nn.LeakyReLU()
        self.act1 = nn.ReLU()
        #self.act1 = nn.Sigmoid()
        #self.act = nn.PReLU()
        #self.act1 = nn.PReLU()
    
    # prediction function
    def forward(self, x):
        #x_ = self.act(self.bn1(self.linear_one(x)))
        #x_ = self.act(self.bn2(self.linear_two(x_)))
        #x_ = self.act(self.bn3(self.linear_three(x_)))
        x_ = self.act(self.linear_one(x))
        #x_ = self.dropout_1(x_)
        x_ = self.act(self.linear_two(x_))
        #x_ = self.dropout_2(x_)
        x_ = self.act(self.linear_three(x_))
        #y = self.act1(self.linear_three(x_))
        #x_ = self.dropout(x_)
        #x_ = self.act(self.bn3(self.linear_three(x_)))
        y = self.act1(self.linear_four(x_))
        #x_ = self.act(self.linear_four(x_))
        #y = self.act1(self.linear_five(x_))
        
        return y

mlp_net = Approx_net(input_size=8, hidden_neurons_1=16, hidden_neurons_2=8, hidden_neurons_3=4, output_size=1)
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
        pointcloud_pred = range_image_to_pointcloud_pytorch(image_pred * kitti_max_distance, device)
        pointcloud_gt = range_image_to_pointcloud_pytorch(image_gt * kitti_max_distance, device)

        chamfer_loss = kaolin.metrics.pointcloud.chamfer_distance(pointcloud_pred[:,:,:3].to(self.device), pointcloud_gt[:,:,:3].to(self.device))
        chamfer_loss_mean = torch.mean(chamfer_loss)

        return chamfer_loss_mean

#loss_fn = ChamferLoss(device=device) #No considera el valor de intensidad para calcular el error, por eso se utiliza MAE
loss_fn = nn.L1Loss()

############################## OPTIMIZER - LR SCHEDULER ###############################

#optimizer = torch.optim.Adam(mlp_net.parameters(), lr=lr)
optimizer = torch.optim.Rprop(mlp_net.parameters(), lr=lr, etas=(0.5,1.5))
#optimizer.param_groups[0]['initial_lr'] = lr
#optimizer = torch.optim.SGD(mlp_net.parameters(), lr=lr)
#optimizer = torch.optim.RMSprop(mlp_net.parameters(), lr=lr)
#optimizer = torch.optim.LBFGS(mlp_net.parameters(), lr=lr, max_iter=5)#, history_size=100)

#optimizer = optim.Apollo(mlp_net.parameters(), lr=lr) #No sirve: val_loss 1.13 despues de 500 epocas
#optimizer = optim.Adahessian(mlp_net.parameters(), lr=lr) #No puedo hacer que converja, ni bajando el lr
#optimizer = optim.Yogi(mlp_net.parameters(), lr=lr) #No converge tanto, llega 4 el error, igual puede verse de cambiar algunos hiperparámetros
#optimizer = optim.QHAdam(mlp_net.parameters(), lr=lr) #Mejor, se deben probar con hiperparámetros lr < 0.1
#optimizer = optim.DiffGrad(mlp_net.parameters(), lr=lr) #Mejor, probar primero de vuelta
#optimizer = optim.AdaBound(mlp_net.parameters(), lr=lr)
#optimizer = torch.optim.RAdam(mlp_net.parameters(), lr=lr, weight_decay=1e-6)


base_optimizer = torch.optim.Rprop(mlp_net.parameters(), lr=lr)
optimizer = optim.Lookahead(base_optimizer, k=5, alpha=0.5)

#lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=5) #Cada 10 épocas lr = lr * gamma
#lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=5, last_epoch=30)
#lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=0.01)
#lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=0.01, max_lr=lr, step_size_up=10, cycle_momentum=False)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=10, T_mult=1, eta_min=0.01)
#iters = len(train_urls) 

################################# WRITER ############################################
def parameters_writer(parameters_list, save_path):
    with open(save_path, 'a+') as f:
        for param in parameters_list:
            f.write(f'{param}, ')
        f.write('\n')

############################## TRAIN LOOP #############################################

def sim_annealing_weights(model, std_percent):
    #params = model.parameters()
    for name, param in model.state_dict().items():
        #print(f'parameters: {param}')
        std_param = (torch.sqrt(torch.mean(torch.square(param)))).item() #RMS de los pesos de cada capa
        std_param = std_param * std_percent #Porcentaje del RMS de los pesos de cada capa
        random_values = torch.empty_like(param).normal_(mean=0.0, std=std_param) #Distribución normal de valores aleatorios para sumar a los pesos
        #print(f'random_values: {random_values}')
        new_param = param + random_values
        model.state_dict()[name].copy_(new_param)
        #param = new_param

#_, hr = next(iter(train_dataloader))
hr = torch.ones((1, 1, high_res_height, high_res_width))
#hr = torch.ones((1, 1, high_res_height, 2048))
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
        lrimgs, hrimgs = data
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
        
        real_pixels = hrimgs[:,:,1:hrimgs.shape[2]-1:2, :hrimgs.shape[3]]

        # Compute the loss and its gradients
        loss = loss_fn(pixels, real_pixels)
        #print(loss)

        # compute penalty only for net.hidden parameters
        #l1_penalty = l1_weight * sum([p.abs().sum() for p in net.hidden.parameters()])
        #l2_penalty = l2_weight * sum([(p**2).sum() for p in mlp_net.parameters()])
        #loss_with_penalty = loss + l2_penalty# + l1_penalty
        #loss_with_penalty.backward()
        loss.backward()
        #loss.backward(create_graph=True)
        
        # Adjust learning weights
        optimizer.step()

        #lr_scheduler.step(epoch_index + j / iters)
        #lr_scheduler.step()
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
            
            #odd_raw_id = torch.arange(1, hrimgs.shape[2]-1, 2)
            real_pixels = hrimgs[:,:,1:hrimgs.shape[2]-1:2, :hrimgs.shape[3]]

            # Compute the loss and its gradients
            loss = loss_fn(pixels, real_pixels)
            #print(loss)
            if loss.requires_grad:
                loss.backward()
    
            return loss
            
        # Adjust learning weights
        optimizer.step(closure)
        #lr_scheduler.step(epoch_index + j / iters)
        
        # calculate the loss again for monitoring
        windows = get_windows(lrimgs, new_pixel_coords)
        pixels = torch.empty((windows.shape[0], windows.shape[1], 1), device=device)
        for i in range(windows.shape[1]):
            pixels[:,i] = mlp_net(windows[:,i,:])
        pixels = torch.flatten(pixels)
        pixels = pixels.view(lrimgs.shape[0], lrimgs.shape[1], -1, lrimgs.shape[-1])  
        real_pixels = hrimgs[:,:,1:hrimgs.shape[2]-1:2, :hrimgs.shape[3]]

        loss = loss_fn(pixels, real_pixels)
        '''        
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
    
    #if ((epoch+1) % 10 == 0):
    #    print(f'Desviación estandar de ruido agregado: {std_percent}')       
    #    sim_annealing_weights(mlp_net, std_percent)
    #    std_percent = std_percent * 0.5

    # We don't need gradients on to do reporting
    running_vloss = 0.0
    mlp_net.eval()

    with torch.no_grad():
        for k, vdata in enumerate(valid_dataloader):
            vlrimgs, vhrimgs = vdata
            vlrimgs, vhrimgs = vlrimgs.to(device), vhrimgs.to(device)
            
            vwindows = get_windows(vlrimgs, new_pixel_coords)
            
            vpixels = torch.zeros((vwindows.shape[0], vwindows.shape[1], 1), device=device)
            for i in range(vwindows.shape[1]):
                vpixels[:,i] = mlp_net(vwindows[:,i,:])
            
            vpixels = torch.flatten(vpixels)
            vpixels = vpixels.view(vlrimgs.shape[0], vlrimgs.shape[1], -1, vlrimgs.shape[-1])

            real_vpixels = vhrimgs[:,:,1:vhrimgs.shape[2]-1:2, :vhrimgs.shape[3]]
            
            #print(f'vpixels: {vpixels.max().item(), vpixels.min().item(), torch.std_mean(vpixels)}')
            #print(f'real_vpixels: {real_vpixels.max().item(), real_vpixels.min().item(), torch.std_mean(real_vpixels)}')

            vloss = loss_fn(vpixels, real_vpixels)
            running_vloss += vloss.item()
            
        avg_vloss = running_vloss / (k + 1)

        #lr_scheduler.step(avg_vloss)
        #lr_scheduler.step()
    fin = time.time()
    print(f'Epoch {epoch_number + 1} - Train_loss: {avg_loss} - Valid_loss: {avg_vloss} - lr: {optimizer.param_groups[0]["lr"]} - Tiempo: {(fin-inicio)/60.0} minutos - Estimado: {((EPOCHS-epoch_number-1)*(fin-inicio))/60.0} minutos')
    
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\otras_arquitecturas\3_pytorch_interpolation\results\mlp_8201061\model_Leaky_ReLU_8201061_30ep_fulldataset_Chamfer\model_Leaky_ReLU_816841_kitti3d_MAE_Rprop_int_ep{epoch_number+1}.pth'
        torch.save(mlp_net.state_dict(), model_path)

    train_parameters.append(f'Epoch {epoch_number + 1} - Train_loss: {avg_loss} - Valid_loss: {avg_vloss} - lr: {optimizer.param_groups[0]["lr"]}')
    parameters_writer(train_parameters, writer_path)

    epoch_number += 1

model_path = rf'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\otras_arquitecturas\3_pytorch_interpolation\results\mlp_8201061\model_Leaky_ReLU_8201061_30ep_fulldataset_Chamfer\model_Leaky_ReLU_816841_kitti3d_MAE_Rprop_int_ep{epoch_number}.pth'
torch.save(mlp_net.state_dict(), model_path)