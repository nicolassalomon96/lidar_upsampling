import torch.nn
from unet_model import *
from data_gen_distance import *
from pointcloud_utils_functions_v2 import *

Unet = UNet()
Unet.load_state_dict(torch.load(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\unet\6_pytorch_implementation\model\Unet_Chamfer_Adam_ep98.pth'))

device = 'cpu'

path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\testing\velodyne\000001.bin'
pointcloud = read_bin(path)
hr_range_distance_image = pointcloud_to_range_image(pointcloud, filter_ego_compensed=True, kind='distance', size=(64,1024)) / kitti_max_distance

downsampling_factor = 4
ind = np.arange(0,64,downsampling_factor)
lr_range_distance_image = hr_range_distance_image[ind] 
input = torch.from_numpy(lr_range_distance_image).view(1,1,lr_range_distance_image.shape[0],lr_range_distance_image.shape[1])
output = torch.from_numpy(hr_range_distance_image).view(1,1,hr_range_distance_image.shape[0],hr_range_distance_image.shape[1])

Unet = Unet.to(device)      

toutputs = Unet(input)

input_image = input.numpy()[0][0]
output_image = toutputs.detach().numpy()[0][0]

pointcloud = range_image_to_pointcloud(output_image * kitti_max_distance)
save_PATH = r'C:\Users\Nicolas\Desktop\Nueva carpeta\pruebas_resultados\b.ply'
save_ply(pointcloud, save_PATH)


