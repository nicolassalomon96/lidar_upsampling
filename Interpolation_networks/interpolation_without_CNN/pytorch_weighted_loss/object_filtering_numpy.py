import sys
import pandas as pd
from pointcloud_utils_functions_v2 import *

def removePoints(PointCloud):
    # Remove the point out of range x,y,z
    BoundaryCond = {"minX": 0, "maxX": 50, "minY": -25, "maxY": 25, "minZ": -2.73, "maxZ": 1.27}

    mask = np.where((PointCloud[:, 0] >= BoundaryCond['minX']) & (PointCloud[:, 0] <= BoundaryCond['maxX']) & (PointCloud[:, 1] >= BoundaryCond['minY']) & (
            PointCloud[:, 1] <= BoundaryCond['maxY']) & (PointCloud[:, 2] >= BoundaryCond['minZ']) & (PointCloud[:, 2] <= BoundaryCond['maxZ']))
    PointCloud = PointCloud[mask]

    return PointCloud


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
    height, width, lenght = dimension
    x_offset = lenght/4 #/ 2
    y_offset = 0.2#height / 4
    z_offset = width/4 #/ 2

    x, y, z = location
    x_corners = [lenght/2 + x_offset, lenght/2 + x_offset, -lenght/2 - x_offset, -lenght/2 - x_offset,  lenght/2 + x_offset,  lenght/2 + x_offset,
                 -lenght/2 - x_offset, -lenght/2 - x_offset]
    y_corners = [y_offset, y_offset, y_offset, y_offset, -height - y_offset, -height - y_offset, -height - y_offset, -height - y_offset]
    z_corners = [width/2 + z_offset, -width/2 - z_offset, -width/2 - z_offset, width/2 + z_offset, width/2 + z_offset, -width/2 - z_offset,
                 -width/2 - z_offset, width/2 + z_offset]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])

    # transform 3d box based on rotation along Y-axis
    R_matrix = np.array([[np.cos(rotation), 0, np.sin(rotation)],
                         [0, 1, 0],
                         [-np.sin(rotation), 0, np.cos(rotation)]])

    corners_3d = np.dot(R_matrix, corners_3d).T

    # shift the corners to from origin to location
    corners_3d = corners_3d + np.array([x, y, z])

    # from camera coordinate to velodyne coordinate
    corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])

    return corners_3d

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
    for i in range(len(label_data.axes[0])):
        dimension = [label_data.iloc[i].height_object, label_data.iloc[i].width_object, label_data.iloc[i].length_object]
        location = [label_data.iloc[i].location_x_camera, label_data.iloc[i].location_y_camera, label_data.iloc[i].location_z_camera]
        rotation = label_data.iloc[i].rotation_y

        corners_3d.append(transform_3dbox_to_pointcloud(dimension, location, rotation))                                                                                          

    return corners_3d

def filter_points(pd_pointcloud, corners):
    #Keep only points inside bbox
    filtered_points = pd_pointcloud[(corners[:,0].min() < pd_pointcloud['X']) & (pd_pointcloud['X'] < corners[:,0].max())
                                      & (corners[:,1].min() < pd_pointcloud['Y']) & (pd_pointcloud['Y'] < corners[:,1].max())
                                      & (corners[:,2].min() < pd_pointcloud['Z']) & (pd_pointcloud['Z'] < corners[:,2].max())]
    return filtered_points

def get_outer_points(pd_pointcloud, filtered_pointcloud):

    # Crear los DataFrames de ejemplo
    df1 = pd_pointcloud.copy()
    df2 = filtered_pointcloud.copy()

    # Realizar un merge con indicator=True
    merged = df1.merge(df2, how='outer', indicator=True)

    # Filtrar las filas que están en ambos DataFrames
    result = merged[merged['_merge'] == 'both']
    # Eliminar las filas repetidas de df1
    df1 = df1[~df1.isin(result.to_dict('list')).all(1)]

    # Eliminar las filas repetidas de df2 (opcional)
    #df2 = df2[~df2.isin(result.to_dict('list')).all(1)]

    return df1

def pointcloud_filter(pointcloud, labels_path):

    pointcloud = removePoints(pointcloud)
    pd_pointcloud = pd.DataFrame(pointcloud, columns=['X', 'Y', 'Z', 'I'])
    
    label_data = pd.read_csv(labels_path, sep=' ', header=None, names=['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
                                                                     'height_object', 'width_object', 'length_object', 'location_x_camera', 'location_y_camera', 
                                                                     'location_z_camera', 'rotation_y'])
    label_data = label_data[label_data['type']!='DontCare']

    corners_all = get_3d_corners(label_data)

    #Eficiente, pero no extrae exactamente los puntos dentro del bbox porque los bbox no estan alineados con el centro del mundo
    for idx, corners in enumerate(corners_all):
        filtered_points = filter_points(pd_pointcloud, corners)
        if idx == 0:
            filtered_pointcloud = filtered_points.copy()
        else:
            filtered_pointcloud = pd.concat([filtered_pointcloud, filtered_points], ignore_index=True)
        
    filtered_pointcloud.reset_index(inplace=True)
    filtered_pointcloud.drop(['index'], axis=1, inplace=True)
    outer_points = get_outer_points(pd_pointcloud, filtered_pointcloud)

    return filtered_pointcloud.to_numpy(), outer_points.to_numpy()
    

if __name__ == "__main__":

    dataset_path = r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kitti\kitti_3d_object\training'
    pointcloud_fullpath = dataset_path + r'\velodyne\000034.bin'
    labels_path = dataset_path + r'\label_2\000034.txt'

    pointcloud = read_bin(pointcloud_fullpath)
    print(pointcloud.shape)
    filtered_pointcloud, non_filtered_pointcloud = pointcloud_filter(pointcloud, labels_path)
    print(filtered_pointcloud.shape)

    #save_path = r'D:\Nicolas\reducida.ply'
    #save_ply(filtered_pointcloud, save_path)

    #save_path_or = r'D:\Nicolas\original.ply'
    #save_ply(pointcloud, save_path_or)

    #save_path_or = r'D:\Nicolas\non_reducida.ply'
    #save_ply(non_filtered_pointcloud, save_path_or)