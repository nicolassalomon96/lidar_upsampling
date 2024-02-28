#Script para filtrar los objetos etiqueteados en una nube de puntos y generar una nueva nube solamente con dichos objetos
import sys
import pandas as pd
import torch
from pointcloud_utils_functions_v2 import *
from config import *

device = 'cuda'

BoundaryCond = {"minX": 0, "maxX": 50, "minY": -25, "maxY": 25, "minZ": -2.73, "maxZ": 1.27}
def removePoints(PointCloud):
    # Remove the point out of range x,y,z
    masks = (
        (PointCloud[:, :, 0] >= BoundaryCond['minX']) & 
        (PointCloud[:, :, 0] <= BoundaryCond['maxX']) & 
        (PointCloud[:, :, 1] >= BoundaryCond['minY']) & 
        (PointCloud[:, :, 1] <= BoundaryCond['maxY']) & 
        (PointCloud[:, :, 2] >= BoundaryCond['minZ']) & 
        (PointCloud[:, :, 2] <= BoundaryCond['maxZ'])
    )

    # Apply the masks to filter the points in each point cloud
    filtered_point_clouds = [PointCloud[mask] for PointCloud, mask in zip(PointCloud, masks)]

    return filtered_point_clouds

def read_labels(labels_path):
    labels = []
    for label_path in labels_path:
        label_data = pd.read_csv(label_path, sep=' ', header=None, names=['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
                                                                            'height_object', 'width_object', 'length_object', 'location_x_camera', 'location_y_camera', 
                                                                            'location_z_camera', 'rotation_y'])
        label_data = label_data[label_data['type']!='DontCare']
        labels.append(label_data.values.tolist())
    return labels

def transform_3dbox_to_pointcloud(dimension, location, rotation):
    """
    #source: https://github.com/HengLan/Visualize-KITTI-Objects-in-Videos/blob/main/utility.py#L45
    # https://github.com/HengLan/Visualize-KITTI-Objects-in-Videos/blob/main/KITTI.py#L269
    convert the 3d box to coordinates in pointcloud
    :param dimension: height, width, and length
    :param location: x, y, and z
    :param rotation: rotation parameter
    :return: transformed coordinates
    """
    height, width, length = dimension
    x_offset = length/4 #torch.tensor(0.0)
    y_offset = torch.tensor(0.4) #El valor 0.4 se colocó a mano. Al generar las imágenes de rango con el algoritmo que elimina las compensaciones de 
                                                     #movimiento de Kitti, la nube de puntos reestablecida esta un poco más arriba en el eje z
    z_offset = width/4 #torch.tensor(0.0)

    x, y, z = location

    x_corners = [length / 2 + x_offset, length / 2 + x_offset, -length / 2 - x_offset, -length / 2 - x_offset,
                              length / 2 + x_offset, length / 2 + x_offset, -length / 2 - x_offset, -length / 2 - x_offset]
    x_corners = torch.vstack(x_corners)
    x_corners = torch.transpose(x_corners, 0, 1) #[num_objects, 8-x_corners]

    y_corners = [torch.zeros_like(height) + y_offset, torch.zeros_like(height) + y_offset, torch.zeros_like(height) + y_offset, torch.zeros_like(height) + y_offset,
                -height - y_offset, -height - y_offset, -height - y_offset, -height - y_offset]
    y_corners = torch.vstack(y_corners)
    y_corners = torch.transpose(y_corners, 0, 1) #[num_objects, 8-y_corners]

    z_corners = [width / 2 + z_offset, -width / 2 - z_offset, -width / 2 - z_offset, width / 2 + z_offset,
                              width / 2 + z_offset, -width / 2 - z_offset, -width / 2 - z_offset, width / 2 + z_offset]
    z_corners = torch.vstack(z_corners)
    z_corners = torch.transpose(z_corners, 0, 1) #[num_objects, 8-z_corners]

    # Create a tensor of corners_3d
    corners_3d = torch.stack([x_corners, y_corners, z_corners], dim=1) #[[x1_corners, y1_corners, z1_corners], [x2_corners, y2_corners, z2_corners], ...] --> [num_objects, 3, 8]

    # transform 3d box based on rotation along Y-axis
    corners_3d_all = []
    for i, rot in enumerate(rotation):

        rotation_matrix_i = torch.tensor([[torch.cos(rot)   , torch.tensor(0.0), torch.sin(rot)],
                            [torch.tensor(0.0), torch.tensor(1.0), torch.tensor(0.0)],
                            [-torch.sin(rot)  , torch.tensor(0.0), torch.cos(rot)]]) #[3, 3]

        corners_3d_i = torch.transpose(torch.mm(rotation_matrix_i, corners_3d[i]), 0, 1) #[8, 3]
        
        # shift the corners to from origin to location
        corners_3d_i = corners_3d_i + torch.tensor([x[i], y[i], z[i]])
        
        # from camera coordinate to velodyne coordinate
        corners_3d_i = corners_3d_i[:, [2, 0, 1]] * torch.tensor([[1.0, -1.0, -1.0]])
        corners_3d_all.append(corners_3d_i)   
    
    stacked_corners = torch.stack(corners_3d_all, dim=0) #[num_objects, 8, 3]

    return stacked_corners

def get_3d_corners(label_data):
    ''' x-axis points to the front
        y-axis points to left
        corners: (8,3) array of vertices for the 3d box in following order [X, Y, Z]:
            6 -------- 7
           /|         /|
          5 -------- 4 .
          | |        | |
          . 0 -------- 1
          |/         |/
          3 -------- 2
    '''
    corners_3d = []
    dimension, location, rotation = [], [], []
    labels_indexes = {'height_object':8, 'width_object':9, 'length_object':10,'location_x_camera':11, 'location_y_camera':12, 'location_z_camera':13, 'rotation_y':14}

    for all_labels in label_data:
        dimension = torch.tensor([[height[labels_indexes['height_object']] for height in all_labels],
                     [width[labels_indexes['width_object']] for width in all_labels],
                     [length[labels_indexes['length_object']] for length in all_labels]]) #[3, num_objects]
        location = torch.tensor([[loc_x[labels_indexes['location_x_camera']] for loc_x in all_labels],
                     [loc_y[labels_indexes['location_y_camera']] for loc_y in all_labels],
                     [loc_z[labels_indexes['location_z_camera']] for loc_z in all_labels]]) #Mismo formato que dimension
        rotation = torch.tensor([rot_y[labels_indexes['rotation_y']] for rot_y in all_labels])

        corners_transformed = transform_3dbox_to_pointcloud(dimension, location, rotation)
        corners_3d.append(corners_transformed)

    return corners_3d

def filter_points(pointcloud, corners_all_objs, normalized_output=False):
    #Keep only points inside bbox
    filtered_points = []

    for corners in corners_all_objs:
        points = pointcloud[(corners[:,0].min() < pointcloud[:,0]) & (pointcloud[:,0] < corners[:,0].max())
                            & (corners[:,1].min() < pointcloud[:,1]) & (pointcloud[:,1] < corners[:,1].max())
                            & (corners[:,2].min() < pointcloud[:,2]) & (pointcloud[:,2] < corners[:,2].max())]
        if normalized_output and len(points):
            points[:,0] = points[:,0] / BoundaryCond['maxX']
            points[:,1] = points[:,1] / BoundaryCond['maxY']
            points[:,3] = points[:,3] / kitti_max_distance
            #points[:,2] = points[:,2] / BoundaryCond['maxZ'] #Como la normalización de Z va a depender si es positivo o negativo, para no 
                                                              #relentizar el código, no se normalizará, ya que de todas formas, Z no toma valores grandes
        
        filtered_points.append(points) #lista en la que cada elemento es una nube de puntos del objecto filtrado
        ###
        #if len(points>0):
        #    filtered_points.append(points) #lista en la que cada elemento es una nube de puntos del objecto filtrado
        #else:
        #    distance_value = torch.linalg.norm(corners[0], 2, axis=0).unsqueeze(0) #sqrt(x^2 + y^2 + z^2)
        #    new_point = torch.cat((corners[0].to(device), distance_value.to(device)))#.requires_grad_() #[x,y,z,dist]
        #    if normalized_output:
        #       #new_point_normalized = torch.cat((new_point[0] / BoundaryCond['maxX'], new_point[1] / BoundaryCond['maxY'], new_point[2], new_point[3] / kitti_max_distance))
        #        new_point[0] = new_point[0] / BoundaryCond['maxX']
        #        new_point[1] = new_point[1] / BoundaryCond['maxY']
        #        new_point[3] = new_point[3] / kitti_max_distance               
        #    new_point.requires_grad_()
        #    filtered_points.append(new_point)
        ###
    filtered_points = torch.vstack(filtered_points) #tensor con todos los puntos referentes a los objetos de una nube, juntos   
    return filtered_points

def get_outer_points(pointcloud, filtered_pointcloud, normalized_output):

    # Broadcast and compare to find matching rows
    matches = (pointcloud[:, None] == filtered_pointcloud).all(dim=-1).any(dim=-1)

    # Use the matches to select rows from main_tensor
    resulting_tensor = pointcloud[~matches]

    if normalized_output and len(resulting_tensor):
        resulting_tensor[:,0] = resulting_tensor[:,0] / BoundaryCond['maxX']
        resulting_tensor[:,1] = resulting_tensor[:,1] / BoundaryCond['maxY']
        resulting_tensor[:,3] = resulting_tensor[:,3] / kitti_max_distance

    return resulting_tensor

def pointcloud_filter(pointcloud, labels_path, normalized_output=False, consider_out_image_points=True):
    
    if not consider_out_image_points:
        pointcloud = removePoints(pointcloud) #lista de pointclouds -->  [batch][num_points, 4] - requires_grad=True

    label_data = read_labels(labels_path) #[batch][num_objects][labels]
    corners_all = get_3d_corners(label_data) #list of tensor --> [batch][num_objects, 8, 3]
    
    #Eficiente, pero no extrae exactamente los puntos dentro del bbox porque los bbox no estan alineados con el centro del mundo
    filtered_pointcloud = []
    non_labeled_pointclouds = []

    for idx, corners in enumerate(corners_all):
        
        filtered_points = filter_points(pointcloud[idx], corners, normalized_output) #Lista donde cada elemento es la nube de puntos filtrada
        #print(idx)
        #if len(filtered_points) > 0: #Comprobación de que luego de removePoints, existen puntos pertenecientes a objetos en la nube
        filtered_pointcloud.append(filtered_points)
        #else: 
            #Si no hay puntos, se hace un padding de zeros para que no hayan errores de dimensiones al calcular la función de error. PERO:
            #Si un punto en ground truth esta en una posición mientras que en la predicha esta en 0, la distancia chamfer, tomará el punto más cercano
            #a la predicha en 0,0 de la ground truth, lo que puede dar un error grande al querer aproximar los puntos de los objetos
        #    filtered_pointcloud.append(torch.zeros((1,4), requires_grad=True))

        non_labeled_points = get_outer_points(pointcloud[idx], filtered_points, normalized_output)
        #if len(non_labeled_points) > 0: #Comprobación de que luego de removePoints, existen puntos pertenecientes a objetos en la nube
        non_labeled_pointclouds.append(non_labeled_points)
        #else:
        #    non_labeled_pointclouds.append(torch.zeros((1,4), requires_grad=True))

    return non_labeled_pointclouds, filtered_pointcloud
    
if __name__ == "__main__":

        dataset_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training'
        labels_path_1 = dataset_path + r'\label_2\001068.txt'

        range_image = np.load(r'.\range_images\original.npy')
        range_image_32 = np.load(r'.\range_images\original_32.npy')
        range_image_inter = np.load(r'.\range_images\interpolada.npy')
       
        range_image_inter[range_image < kitti_carla_min_range] = 0.0
        range_image_inter[range_image > kitti_max_distance] = 0.0
        
        range_image_inter = torch.from_numpy(range_image_inter).unsqueeze(0) #[batch, channel, height, width]
        range_image_inter = range_image_inter.unsqueeze(-1)
        range_image = torch.from_numpy(range_image).unsqueeze(0)
        range_image = range_image.unsqueeze(-1)
        range_image_32 = torch.from_numpy(range_image_32).unsqueeze(0)
        range_image_32 = range_image_32.unsqueeze(0)
        range_image_64 = range_image.unsqueeze(0)

        range_image_batch = torch.stack((range_image, range_image_inter), dim=0) #[batch, channels, height, width] [2,1,64,2048]
        labels_path = [labels_path_1, labels_path_1]
        labels_path_32 = [labels_path_1]
        ############################### HASTA AQUI ESTAMOS EN EL MUNDO DE LAS IMÁGENES#####################################

        pointcloud_batch = range_image_to_pointcloud_pytorch(range_image_batch.to(device), device='cuda') #[batch, num_points, 4]
        pointcloud_32 = range_image_to_pointcloud_pytorch(range_image_32.to(device), device='cuda') #[batch, num_points, 4]
        pointcloud_64 = range_image_to_pointcloud_pytorch(range_image_64.to(device), device='cuda')
        
        non_filtered_points, filtered_points = pointcloud_filter(pointcloud_batch, labels_path, normalized_output=False)
        non_filtered_points_32, filtered_points_32 = pointcloud_filter(pointcloud_32, labels_path_32, normalized_output=False)

        save_path_or_64 = r'.\pointclouds_32to64\original_64.ply'
        save_ply(pointcloud_64[0].detach().cpu().numpy(), save_path_or_64)

        save_path_or = r'.\pointclouds_32to64\puntos_originales_filtrados.ply'
        save_ply(filtered_points[0].detach().cpu().numpy(), save_path_or)

        save_path_red = r'.\pointclouds_32to64\puntos_reducidos_filtrados.ply'
        save_ply(filtered_points_32[0].detach().cpu().numpy(), save_path_red)
        
        save_path = r'.\pointclouds_32to64\puntos_interpolados_filtrados.ply'
        save_ply(filtered_points[1].detach().cpu().numpy(), save_path)



