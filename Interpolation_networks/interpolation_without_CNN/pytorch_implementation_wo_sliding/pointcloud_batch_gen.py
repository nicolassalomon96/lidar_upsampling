import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pointcloud_utils_functions_v2 import *
#from data_gen import *
from data_gen_all import *
device = "cuda" if torch.cuda.is_available() else "cpu"

#pointcloud_saved_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Complex-YOLO\dataset\kitti\training\velodyne_64_mi_red_nuevos'
pointcloud_saved_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Complex-YOLO\dataset\kitti\testing\velodyne_64_mired_616841_dataset_paper'

if not os.path.exists(pointcloud_saved_path):
    os.mkdir(pointcloud_saved_path)

############################################## RED DE INTERPOLACIÓN ENTRENADA ####################################################
#saved_model_path = r'.\models_result\model_816841_dataset_paper_sin_CNN_ep239.pth' #Mi modelo entrenado con dataset generado con técnica del paper
saved_model_path = r'.\models_result\best_616841_0.0038.pth' #Mi modelo (6-16-8-4-1) entrenado con dataset generado con técnica del paper y MSE

class Approx_net(nn.Module):    
    def __init__(self, input_size, hidden_neurons_1, hidden_neurons_2, hidden_neurons_3, output_size):
        super(Approx_net, self).__init__()
        # hidden layer 
        self.linear_one = nn.Linear(input_size, hidden_neurons_1, bias=True)
        self.linear_two = nn.Linear(hidden_neurons_1, hidden_neurons_2, bias=True)
        self.linear_three = nn.Linear(hidden_neurons_2, hidden_neurons_3, bias=True)
        self.linear_four = nn.Linear(hidden_neurons_3, output_size, bias=True)

        self.act = nn.LeakyReLU()
        self.act1 = nn.ReLU()
    
    # prediction function
    def forward(self, x):
        x_ = self.act(self.linear_one(x))
        x_ = self.act(self.linear_two(x_))
        x_ = self.act(self.linear_three(x_))
        y = self.act1(self.linear_four(x_))
        return y

model = Approx_net(input_size=6, hidden_neurons_1=16, hidden_neurons_2=8, hidden_neurons_3=4, output_size=1)
model.load_state_dict(torch.load(saved_model_path))
model.to(device)
################################################ GENERACIÓN DE DATASET ######################################################
class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()
    
train_dataset = IterDataset(data_generator)

batch_size = 200
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

################################################ CREACIÓN DEL MODELO Y PROCESADO DE DATOS ####################################
def interpolate(input, mlp_net):
    #hr = torch.ones((1, 1, 64, 2048))
    #row_pos, column_pos = torch.where(hr[0,0] >= 0.0)
    #odd_row_pos = row_pos[row_pos % 2 != 0] / row_pos.max()
    #column_pos = column_pos[:column_pos.shape[0]//2] / column_pos.max()
    #new_pixel_coords = torch.Tensor(list(zip(odd_row_pos, column_pos))[:-2048]) #[:-1024] #Posiciones (x,y) de cada nuevo pixel a interpolar

    #def get_windows(x, new_pixel_coords, upscaling_factor=2):
    def get_windows(x, upscaling_factor=2):
        kernel_size = (2,3)
        padding = (0,1)
        stride = (1,1)
        #new_pixel_coords_batch = new_pixel_coords.unsqueeze(0).repeat(x.shape[0],1,1).to(device)
        
        if upscaling_factor == 2:
            windows = F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)
            windows = windows.transpose(1, 2) #Obtener los valores de la ventana o kernel ordenados por fila, donde cada fila representa una ventana serializada
            #windows = torch.dstack((windows, new_pixel_coords_batch))  
        else:
            print("ERROR: Wrong Upsampling factor")
        return windows  

    #windows = get_windows(input, new_pixel_coords)   
    windows = get_windows(input)
    windows_stack = windows.reshape(windows.shape[0] * windows.shape[1], windows.shape[2])
    pixels = mlp_net(windows_stack)#.squeeze()

    #pixels = torch.empty((twindows.shape[0], twindows.shape[1], 1), device=device)
    #for i in range(twindows.shape[1]):
    #    pixels[:,i] = mlp_net(twindows[:,i,:])

    #pixels = torch.flatten(pixels)
    pixels = pixels.view(input.shape[0], input.shape[1], -1, input.shape[-1])

    toutputs = torch.zeros((input.shape[0], 1, 64, 2048), device=device, dtype=torch.double)
    toutputs[:,:,:toutputs.shape[2]:2, :toutputs.shape[3]] = input
    toutputs[:,:,1:toutputs.shape[2]-1:2, :toutputs.shape[3]] = pixels
    toutputs = torch.dstack((toutputs[:,:,:-1,:], input[:,:,-1:,:])) #Repito la última fila
    
    return toutputs

pointcloud_number = 0
for i, data in enumerate(tqdm(dataloader)):
    lrimg_distance, hrimg_distance, lrimg_intensity, hrimg_intensity = data
    lrimg_distance = lrimg_distance.to(device)
    lrimg_intensity = lrimg_intensity.to(device)

    ###################################### PREDICCIÓN DE PIXELES DE DISTANCIA #########################################
    outputs_distance = interpolate(lrimg_distance, model)
    pointcloud_gen_batch_distance = range_image_to_pointcloud_pytorch(outputs_distance * kitti_max_distance, device)
    ###################################### PREDICCIÓN DE PIXELES DE INTENSIDAD #########################################
    outputs_intensity = interpolate(lrimg_intensity, model)

    for j, pointcloud in enumerate(pointcloud_gen_batch_distance):
        pointcloud_join = torch.hstack([pointcloud[:,:3], outputs_intensity[j,0,:,:].reshape(-1,1)])
        save_path = os.path.join(pointcloud_saved_path, f'{str(pointcloud_number).zfill(6)}.bin')
        save_bin(pointcloud_join.cpu().detach().numpy(), save_path)
        pointcloud_number += 1