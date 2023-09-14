a = 1
import torch.nn as nn
import torch.nn.functional as F
from data_gen_distance import *
from pointcloud_utils_functions_v2 import *

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
        nn.init.kaiming_normal_(self.linear_four.weight, nonlinearity='relu')
       
        self.act = nn.LeakyReLU()
        self.act1 = nn.ReLU()
    
    # prediction function
    def forward(self, x):
        x_ = self.act(self.linear_one(x))
        x_ = self.act(self.linear_two(x_))
        x_ = self.act(self.linear_three(x_))
        y = self.act1(self.linear_four(x_))       
        return y

def get_windows(x, new_pixel_coords, upscaling_factor=2):
    kernel_size = (2,3)
    padding = (0,1)
    stride = (1,1)
    
    if upscaling_factor == 2:
        windows = F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)
        windows = windows.transpose(1, 2) #Obtener los valores de la ventana o kernel ordenados por fila, donde cada fila representa una ventana serializada
        windows = windows.to(device)
        windows = torch.dstack((windows, new_pixel_coords))  
    else:
        print("ERROR: Wrong Upsampling factor")
    return windows 

device = 'cpu'
mlp_net = Approx_net(input_size=8, hidden_neurons_1=16, hidden_neurons_2=8, hidden_neurons_3=4, output_size=1)
mlp_net.load_state_dict(torch.load(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\lidar_upsampling\Interpolation_networks\interpolation_without_CNN\pytorch_weighted_loss\trained_models\model_816841_kitti3d_Chamfer_Rprop_ep328_1image.pth'))
mlp_net = mlp_net.to(device)   

pointcloud_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\testing\velodyne\000001.bin'
pointcloud = read_bin(pointcloud_path)
hr_range_distance_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=True, kind='distance', size=(64,1024)) / kitti_max_distance

downsampling_factor = 2
ind = np.arange(0,64,downsampling_factor)
lr_range_distance_image = hr_range_distance_image[ind] 
input = torch.from_numpy(lr_range_distance_image).view(1,1,lr_range_distance_image.shape[0],lr_range_distance_image.shape[1])
output = torch.from_numpy(hr_range_distance_image).view(1,1,hr_range_distance_image.shape[0],hr_range_distance_image.shape[1])

row_pos,column_pos = torch.where(output[0,0] >= 0)
odd_row_pos = row_pos[row_pos % 2 != 0] / row_pos.max()
column_pos = column_pos[:column_pos.shape[0]//2] / column_pos.max()
new_pixel_coords = torch.Tensor(list(zip(odd_row_pos, column_pos))[:-1024])
new_pixel_coords = new_pixel_coords.unsqueeze(0).to(device)

twindows = get_windows(input, new_pixel_coords)
   
pixels = torch.empty((twindows.shape[0], twindows.shape[1], 1), device=device)
for i in range(twindows.shape[1]):
    pixels[:,i] = mlp_net(twindows[:,i,:])

pixels = torch.flatten(pixels)
pixels = pixels.view(input.shape[0], input.shape[1], -1, input.shape[-1])

toutputs = torch.zeros_like(output, device=device, dtype=torch.double)
toutputs[:,:,:toutputs.shape[2]:2, :toutputs.shape[3]] = input
toutputs[:,:,1:toutputs.shape[2]-1:2, :toutputs.shape[3]] = pixels
toutputs = torch.dstack((toutputs[:,:,:-1,:], input[:,:,-1:,:])) #Repito la última fila     

input_image = input.numpy()[0][0]
output_image = toutputs.detach().numpy()[0][0]

#fig, axs = plt.subplots(2,1, figsize=(15,8))
#axs[0].imshow(hr_range_distance_image, cmap='jet')
#axs[1].imshow(output_image, cmap='jet')
#display_range_image(output_image)
#plt.show()

pointcloud = range_image_to_pointcloud(output_image * kitti_max_distance)
#pointcloud = range_image_to_pointcloud(hr_range_distance_image * kitti_max_distance)
save_PATH = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\lidar_upsampling\Interpolation_networks\interpolation_without_CNN\pytorch_weighted_loss\test_outputs\test.ply'
save_ply(pointcloud, save_PATH)

#original = read_bin(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\testing\velodyne\000001.bin')
#original[original[:,2] < -(lidar_z_pos+1.0)] = 0.0
#print(original[:,2].min())
#save_ply(original, save_PATH)