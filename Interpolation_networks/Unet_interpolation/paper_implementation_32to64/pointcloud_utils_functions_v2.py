import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import imageio.v2 as io
from pyntcloud import PyntCloud
#import open3d as o3d
import os
import pandas as pd
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
from config import *

############################################# FUNCIONES PARA PROCESAR NUBES DE PUNTOS E IMÁGENES DE RANGO #############################################

#Leer pointcloud y devolver range image de tipo distancia o intensidad: cada pixel tiene el valor de la distancia o intensidad a dicho punto (Versión 1)
def pointcloud_to_range_image(pointcloud, v_res=0.42, h_res=0.35, v_fov=(-24.9, 2.0), h_fov=(-180,180), size=(64,1024), lidar_16_ch=False, format='kitti', filter_ego_compensed=True, kind='distance', return_angles=False):
    '''
    pointcloud: pointcloud [x,y,z,intensity] --> kitti format!!!
    v_res: vertical resolution
    h_res: horizontal resolution
    v_fov: vertical field of view (ej: (-24.9,2.0))
    h_fov: horizontal field of view (ej: (-180,180))
    lidar_16_ch: imagen resultante de un lidar de 16 canales
    format: "kitti" o "carla" pointcloud. Kitti tiene el sistema de coordenadas inverso en el eje y a CARLA
    kind: tipo de dato a almacenar en cada pixel --> 'distance': distancia, 'intensity': intensidad, 'both':distancia,intensidad, 'all':distancia,intensidad,x,y,z
    filter_ego_compensed: True-->Técnica de filtado para obtener una imagen de rango más limpia
    return_angles: True--> Devuelve los angulos de yaw y pitch
    '''
    def in_h_range_points(m, n, fov):
        #extract horizontal in-range points
        return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), np.arctan2(n, m) < (-fov[0] * np.pi / 180))

    def in_v_range_points(m, n, fov):
        #extract vertical in-range points
        return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), np.arctan2(n, m) > (fov[0] * np.pi / 180))

    def fov_setting(points, x, y, z, dist, h_fov, v_fov):
        #filter points based on h,v FOV
        if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
            return points
        if h_fov[1] == 180 and h_fov[0] == -180:
            return points[in_v_range_points(dist, z, v_fov)]
        elif v_fov[1] == 2.0 and v_fov[0] == -24.9:
            return points[in_h_range_points(x, y, h_fov)]
        else:
            h_points = in_h_range_points(x, y, h_fov)
            v_points = in_v_range_points(dist, z, v_fov)
            return points[np.logical_and(h_points, v_points)]

    proj_H = size[0]
    #proj_H = int(np.ceil(v_fov[1] - v_fov[0]) / v_res) #Range Image Height teniendo en cuenta la resolución vertical de un LIDAR de 64 canales
    #if lidar_16_ch:
    #    proj_H = 16 #Para considerar un LIDAR de 16 canales simulado, de igual resolución vertical al LIDAR de 64 canales

    #proj_W = int(np.ceil((h_fov[1] - h_fov[0]) / h_res)) #Range Image Width
    #proj_W = 1024
    proj_W = size[1]
  
    #lidar field of view
    fov_up = v_fov[1] / 180.0 * np.pi  # field of view up in radians
    fov_down = v_fov[0] / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # total field of view in radians
  
    #scan components
    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]

    dist = np.linalg.norm(pointcloud[:, :3], 2, axis=1) #sqrt(x^2 + y^2 + z^2)

    #angles of all points
    if format == 'kitti':
        yaw = np.arctan2(-y, x) #Angulo de yaw, tomando el frente del auto como 0º (eje x positivo apunta al frente, eje y positivo apunta a la izquierda, por eso
                                #se coloca el signo menos)                                
                               
    elif format == 'carla':
        yaw = np.arctan2(y, x) #En CARLA el eje y positivo apunta a la derecha (al revés que kitti)
    
    elif format == 'nuscene':
        yaw = np.arctan2(x, y) #En Nuscene el eje y positivo apunta a adelante y el eje x hacia la derecha (al revés que kitti) 
    
    pitch = np.arcsin(z / dist)

    #customized field of view
    #yaw = fov_setting(yaw, x, y, z, dist, h_fov, v_fov)
    #pitch = fov_setting(pitch, x, y, z, dist, h_fov, v_fov)
    #dist = fov_setting(dist, x, y, z, dist, h_fov, v_fov)

    #projections in image coords normalized
    if h_fov[0] > 0: 
        proj_x = (yaw - h_fov[0] * np.pi/180)/((h_fov[1]-h_fov[0]) * np.pi/180) 
    else:
        proj_x = (yaw + abs(h_fov[0] * np.pi/180))/((abs(h_fov[0]) + h_fov[1]) * np.pi/180) #0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        #proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]

    #scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]

    # round and clamp for use as index
    proj_x = (np.floor(proj_x)).astype(int)

    if filter_ego_compensed == True and format == 'kitti':
        threshold = -0.005
        yaw_flipped = -np.arctan2(y, -x)
        yaw_diffs = yaw_flipped[1:] - yaw_flipped[:-1]

        #print(yaw_flipped[1:])
        #print(yaw_flipped[:-1])
        #print(yaw_diffs)

        jump_mask = np.greater(threshold, yaw_diffs) #Cuando la diferencia entre dos ángulos de azimuth consecutivos sea mayor que un cierto umbral, significa que es un nuevo canal 
        ind = np.add(np.where(jump_mask), 1) #np.where retorna la posición de los valores en True, luego les suma 1 a cada posición. Me dice donde comienza una nueva fila de escaneo del lidar
        #print(np.where(jump_mask))
        
        rows = np.zeros_like(x)
        rows[ind] += 1 #Coloca un 1 en las posiciones donde comienza una nueva fila de escaneo del lidar, luego con np.cumsum irá almacenando el número de canal correcto
        proj_y = np.int32(np.cumsum(rows, axis=-1))
    else:
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
        #proj_y = (fov_up - pitch) / fov # in [0.0, 1.0]
        #proj_y = np.round(proj_y,3) #Redondeo a 3 decimales para que luego los canales (filas) no excedan los proj_H
        #proj_y = np.round(proj_y)

        #scale to image size using angular resolution
        proj_y *= proj_H  # in [0.0, H]
      
        # round and clamp for use as index
        proj_y = (np.floor(proj_y)).astype(int)

    proj_y[proj_y > proj_H - 1] = proj_H - 1 #Ajuste para tener exactamente 64 canales (por el redondeo puedo tener más de 64 filas de datos)
    proj_y[proj_y < 0.0] = 0.0
    proj_x[proj_x > proj_W - 1] = proj_W - 1
    proj_x[proj_x < 0.0] = 0.0  

    range_image = np.zeros((proj_H, proj_W), dtype=np.float32)
    range_image_dist = np.zeros((proj_H, proj_W), dtype=np.float32)
    range_image_intensity = np.zeros((proj_H, proj_W), dtype=np.float32)
    range_image_x = np.zeros((proj_H, proj_W), dtype=np.float32)
    range_image_y = np.zeros((proj_H, proj_W), dtype=np.float32)
    range_image_z = np.zeros((proj_H, proj_W), dtype=np.float32)
    
    #Se ordenan los puntos en orden decrecientes para que en caso de que varios puntos ocupen el mismo pixel, se quede con el valor más pequeño, que sería la distancia mas pequeña
    #o en otras palabras, el punto más cercano
    indices = np.arange(dist.shape[0])
    order = np.argsort(dist)[::-1]
    indices = indices[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    if kind == 'distance':
        range_image[proj_y, proj_x] = dist[order]
    elif kind == 'intensity':
        intensity = pointcloud[:, 3]
        range_image[proj_y, proj_x] = intensity[order]
    elif kind == 'both':
        intensity = pointcloud[:, 3]
        range_image_dist[proj_y, proj_x] = dist[order]
        range_image_intensity[proj_y, proj_x] = intensity[order]
        range_image = np.dstack([range_image_dist, range_image_intensity])
    elif kind == 'all':
        intensity = pointcloud[:, 3]
        range_image_dist[proj_y, proj_x] = dist[order]
        range_image_intensity[proj_y, proj_x] = intensity[order]
        range_image_x[proj_y, proj_x] = x[order]
        range_image_y[proj_y, proj_x] = y[order]
        range_image_z[proj_y, proj_x] = z[order]
        range_image = np.dstack([range_image_dist, range_image_intensity, range_image_x, range_image_y, range_image_z])
        range_image = np.transpose(range_image, (2, 0, 1)) #Formato [bs, 5, filas, columnas]

    else:
        print("Tipo incorrecto")

    if return_angles:
        return range_image, yaw, pitch
    else:
        return range_image

#Leer pointcloud y devolver range image de tipo distancia o intensidad: cada pixel tiene el valor de la distancia o intensidad a dicho punto (Versión 2)
#MAL: Se corrigió un error al calcular los angulos de yaw (al graficarlos se veía que no se consideraba la vuelta completa del lidar). En realidad no es un error
#si graficás la nube de puntos de CARLA con la misma función, da bien, es un error propio del preprocesamiento de kitti
def pointcloud_to_range_image_v2(pointcloud, v_res=0.42, h_res=0.35, v_fov=(-24.9, 2.0), h_fov=(-180,180), lidar_16_ch=False, format='kitti', kind='distance', return_angles=False):
    '''
    pointcloud: pointcloud [x,y,z,intensity] --> kitti format!!!
    v_res: vertical resolution
    h_res: horizontal resolution
    v_fov: vertical field of view (ej: (-24.9,2.0))
    h_fov: horizontal field of view (ej: (-180,180))
    lidar_16_ch: imagen resultante de un lidar de 16 canales
    format: "kitti" o "carla" pointcloud (el lidar gira en sentido contrario a otro, por ello es necesario especificarlo para evitar una imagen de rango espejada)
    kind: tipo de dato a almacenar en cada pixel --> 'distance': distancia, 'intensity': intensidad
    '''
    def in_h_range_points(m, n, fov):
        #extract horizontal in-range points
        return np.logical_and(np.arctan2(n, m) > (-fov[1] * np.pi / 180), np.arctan2(n, m) < (-fov[0] * np.pi / 180))

    def in_v_range_points(m, n, fov):
        #extract vertical in-range points
        return np.logical_and(np.arctan2(n, m) < (fov[1] * np.pi / 180), np.arctan2(n, m) > (fov[0] * np.pi / 180))

    def fov_setting(points, x, y, z, dist, h_fov, v_fov):
        #filter points based on h,v FOV
        if h_fov[1] == 180 and h_fov[0] == -180 and v_fov[1] == 2.0 and v_fov[0] == -24.9:
            return points
        if h_fov[1] == 180 and h_fov[0] == -180:
            return points[in_v_range_points(dist, z, v_fov)]
        elif v_fov[1] == 2.0 and v_fov[0] == -24.9:
            return points[in_h_range_points(x, y, h_fov)]
        else:
            h_points = in_h_range_points(x, y, h_fov)
            v_points = in_v_range_points(dist, z, v_fov)
            return points[np.logical_and(h_points, v_points)]

    proj_H = int(np.ceil(v_fov[1] - v_fov[0]) / v_res) #Range Image Height teniendo en cuenta la resolución vertical de un LIDAR de 64 canales
    if lidar_16_ch:
        proj_H = 16 #Para considerar un LIDAR de 16 canales simulado, de igual resolución vertical al LIDAR de 64 canales

    #proj_W = int(np.ceil((h_fov[1] - h_fov[0]) / h_res)) #Range Image Width
    proj_W = 2048 #1024
  
    #lidar field of view
    fov_up = v_fov[1] / 180.0 * np.pi  # field of view up in radians
    fov_down = v_fov[0] / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # total field of view in radians
  
    #scan components
    x = pointcloud[:, 0]
    y = pointcloud[:, 1]
    z = pointcloud[:, 2]

    dist = np.linalg.norm(pointcloud[:, :3], 2, axis=1) #sqrt(x^2 + y^2 + z^2)
    intensity = pointcloud[:, 3]

    #angles of all points
    if format == 'kitti':
        yaw = np.arctan2(x, y) #Angulo de yaw, tomando el frente del auto como 0º (eje x positivo apunta al frente, eje y positivo apunta a la izquierda)
                               
    elif format == 'carla':
        yaw = np.arctan2(x, -y) #Cambie el y por -y porque en CARLA el eje-y positivo apunta a la derecha y en kitti apunta hacia la izquierda
    
    pitch = np.arcsin(z / dist)

    #customized field of view
    yaw = fov_setting(yaw, x, y, z, dist, h_fov, v_fov)
    pitch = fov_setting(pitch, x, y, z, dist, h_fov, v_fov)
    dist = fov_setting(dist, x, y, z, dist, h_fov, v_fov)

    #projections in image coords normalized
    if h_fov[0] > 0: 
        proj_x = (yaw - h_fov[0] * np.pi/180)/((h_fov[1]-h_fov[0]) * np.pi/180) 
    else:
        proj_x = (yaw + abs(h_fov[0] * np.pi/180))/((abs(h_fov[0]) + h_fov[1]) * np.pi/180) #0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        #proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]

    #print("min_pitch",np.min(pitch))
    #print("max_pitch",np.max(pitch))
    #print("fov_down", fov_down) 
    #print("fov_up", fov_up)
    #print("fov", fov)  

    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
    #proj_y = (fov_up - pitch) / fov # in [0.0, 1.0]

    #proj_y = np.round(proj_y,3) #Redondeo a 3 decimales para que luego los canales (filas) no excedan los proj_H
    #proj_y = np.round(proj_y)
      
    #scale to image size using angular resolution
    proj_x *= proj_W  # in [0.0, W]
    proj_y *= proj_H  # in [0.0, H]
 
    # round and clamp for use as index
    proj_x = (np.floor(proj_x)).astype(int)
    proj_y = (np.floor(proj_y)).astype(int)

    proj_y[proj_y > proj_H - 1] = proj_H - 1 #Ajuste para tener exactamente 64 canales (por el redondeo puedo tener más de 64 filas de datos)
    proj_y[proj_y < 0.0] = 0.0
    proj_x[proj_x > proj_W - 1] = proj_W - 1
    proj_x[proj_x < 0.0] = 0.0  

    range_image = np.zeros((proj_H, proj_W), dtype=np.float32)

    if kind == 'distance':
        range_image[proj_y, proj_x] = dist
    elif kind == 'intensity':
        range_image[proj_y, proj_x] = intensity
    else:
        print("Tipo incorrecto")

    #Shifteo de la imagen 90 grados a la izquierda para coincidir con la nube de puntos original
    range_image_aux = np.append(range_image, range_image, axis=1)
    range_image_shift = range_image_aux[:,range_image.shape[1]//4:range_image.shape[1] + range_image.shape[1]//4]

    if return_angles:
        return range_image_shift, yaw, pitch
    else:
        return range_image_shift

#Leer range image y devolver pointcloud (formato: x,y,z,distancia o intensidad (dependiendo del tipo de imagen de rango))
def range_image_to_pointcloud(image, v_fov=(-24.9, 2.0), h_fov=(-180, 180)):
    '''
    image: range image en formato 0 - max_distance
    v_fov: vertical field of view (ej: (-24.9,2.0))
    h_fov: horizontal field of view (ej: (-180,180))
    '''
    H = image.shape[0]
    W = image.shape[1]
    dist = image.reshape(1,-1)[0]

    dist[dist < kitti_carla_min_range] = 0.0

    proj_x = np.ones((H,1)) * np.arange(0,W,1)  # matriz (W x H) donde cada columna tiene el valor del indice de dicha columna
    proj_x = proj_x.reshape(1,-1)[0]
    proj_y = (np.transpose(np.ones((1,H)) * np.arange(0,H,1)) * np.ones((1,W))) # matriz (W x H) donde cada fila tiene el valor del indice de dicha fila
    proj_y = proj_y.reshape(1,-1)[0]

    fov_up = v_fov[1] / 180.0 * np.pi  # field of view up in radians
    fov_down = v_fov[0] / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # total field of view in radians

    #Desnormalizamos los puntos para obtener los valores de yaw (azimuth) y pitch (elevación)
    proj_x = proj_x/W
    proj_y = proj_y/H

    if h_fov[0] > 0:
        yaw = -(np.pi/180) * (proj_x * (h_fov[1] - h_fov[0]) + h_fov[0]) #Despejo de la ecuación obtenida en la normalización en la función de pointcloud to range image
    else:
        yaw = -(np.pi/180) * (proj_x * abs((h_fov[1] - h_fov[0])) - abs(h_fov[0]))

    pitch = fov * (1 - proj_y) - abs(fov_down)

    x = dist * np.cos(pitch) * np.cos(yaw)
    y = dist * np.cos(pitch) * np.sin(yaw)
    z = dist * np.sin(pitch)
   
    pointcloud = np.dstack((x,y,z,dist))
    pointcloud[pointcloud[:,:,2] < -(lidar_z_pos+lidar_z_offset)] = 0.0 #Filtro los puntos que estan por debajo del suelo más un offset manual

    return pointcloud[0]

#Leer range image y devolver pointcloud (formato: x,y,z,distancia o intensidad (dependiendo del tipo de imagen de rango)) --> usando batch de data con pytorch
def range_image_to_pointcloud_pytorch(tensor_image, device, v_fov=(-24.9, 2.0), h_fov=(-180, 180)):
    #tensor_image: [batch, channel, height, width]
    H = tensor_image.shape[2]
    W = tensor_image.shape[3]
    batch_size = tensor_image.shape[0]
    dist = tensor_image.reshape(batch_size, -1)

    dist[dist < kitti_carla_min_range] = 0.0

    proj_x = torch.ones((batch_size, H, 1)) * torch.arange(0,W,1) # matriz (W x H) donde cada columna tiene el valor del indice de dicha columna
    proj_x = proj_x.reshape((batch_size, -1))
    proj_y = torch.transpose(torch.ones((batch_size,1,H)) * torch.arange(0,H,1), 1, 2) * torch.ones((batch_size,1,W)) # matriz (W x H) donde cada fila tiene el valor del indice de dicha fila
    proj_y = proj_y.reshape((batch_size,-1))

    fov_up = v_fov[1] / 180.0 * torch.pi  # field of view up in radians
    fov_down = v_fov[0] / 180.0 * torch.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # total field of view in radians

    #Desnormalizamos los puntos para obtener los valores de yaw (azimuth) y pitch (elevación)
    proj_x = proj_x/W
    proj_y = proj_y/H

    #print(proj_x.shape)

    if h_fov[0] > 0:
        yaw = -(torch.pi/180) * (proj_x * (h_fov[1] - h_fov[0]) + h_fov[0]) #Despejo de la ecuación obtenida en la normalización en la función de pointcloud to range image
    else:
        yaw = -(torch.pi/180) * (proj_x * abs((h_fov[1] - h_fov[0])) - abs(h_fov[0]))

    pitch = fov * (1 - proj_y) - abs(fov_down)

    #pitch.requires_grad_()
    #yaw.requires_grad_()
    #dist.requires_grad_()

    pitch = pitch.to(device)
    yaw = yaw.to(device)

    x = dist * torch.cos(pitch) * torch.cos(yaw)
    y = dist * torch.cos(pitch) * torch.sin(yaw)
    z = dist * torch.sin(pitch)

    pointcloud = torch.dstack((x,y,z,dist))
    pointcloud[pointcloud[:,:,2] < -(lidar_z_pos+lidar_z_offset)] = 0.0 #Filtro los puntos que estan por debajo del suelo más un offset manual
    return pointcloud

#Función para tomar una imagen de rango de distancia (formato: 0 - max_distance) y de intensidad y devolver una nube de puntos de la forma (X,Y,Z,intensidad)
def range_image_to_pointcloud_with_instensity(range_image_distance, range_image_intensity):
    aux_pointcloud = range_image_to_pointcloud(range_image_distance)
    intensities_serial = range_image_intensity.reshape(-1,1)
    pointcloud = np.hstack([aux_pointcloud[:,:3], intensities_serial])
    return pointcloud

#Aplicar Monte Carlo Dropout junto con filtro de ruido considerando el desvío estandar
def MC_dropout_with_noise_red (lr_image, model, iterations=10, noise_factor=0.03):
    
    gen_images = []
    for _ in range(iterations):
        #gen_images.append(model.predict(lr_image, verbose=0)[0])
         #A pesar que training esta en False, en la clase de MonteCarloDropout training sigue en True, para dar la aleatoreidad necesaria al calculo de la media y desviación
        gen_images.append(model(np.array([lr_image]), training=False)[0])
    
    if iterations > 1:
        gen_image_mean = np.array(gen_images).squeeze().mean(axis=0)
        gen_image_std = np.array(gen_images).squeeze().std(axis=0)

        gen_image_mean[gen_image_std > gen_image_mean * noise_factor] = 0 #remoción del ruido comparándolo con el desvío estandar
        #print(gen_image_mean.shape)
        #print(gen_image_std.shape)
        return gen_image_mean

    elif iterations == 1:
        return np.array(gen_images)[0].squeeze()
    
    else:
        print("Número de iteraciones inválido")    

#Función para hacer un downsamping (por un determinado factor) de todas las imágenes de rango en una carpeta y guardarlas en otra carpeta
def image_downsampling_batch(full_hr_folder_PATH, full_lr_folder_PATH, downsampling_factor, hr_images_row=64):
    #hr_folder_PATH = Path completo al directorio con las imágenes de rango de alta resolución
    #lr_folder_PATH = Path completo al directorio donde se guardarán las imágenes de rango de baja resolución

    if not os.path.exists(full_lr_folder_PATH):
        os.makedirs(full_lr_folder_PATH)
    
    counter = 0
    indexes = range(0, hr_images_row, downsampling_factor)
    images_64ch_path = os.listdir(full_hr_folder_PATH)
    for image_path in images_64ch_path:
        if image_path.endswith('.tif'):
            high_res_image = io.imread(os.path.join(full_hr_folder_PATH, image_path))
            low_res_image = high_res_image[indexes]
            filename = os.path.join(full_lr_folder_PATH, image_path)
            io.imsave(filename, low_res_image)
            counter += 1

    print(f"Se generaron {counter} imágenes de {hr_images_row / downsampling_factor} canales")


############################################# FUNCIONES PARA LEER Y MOSTRAR NUBES DE PUNTOS E IMAGEN DE RANGO #############################################
#Leer pointcloud en formato .bin
def read_bin(bin_path):
    pointcloud = np.fromfile(bin_path, dtype=np.float32, count=-1).reshape([-1,4]) #Lee el .bin con la nube de puntos y lo lleva a la forma [X,Y,Z,I]
    return pointcloud

#Leer pointcloud en formato .ply
def read_ply(ply_path, intensity_channel=False):
    full_pointcloud = PyntCloud.from_file(ply_path)
    pointcloud = full_pointcloud.points.iloc[:,[0,1,2]]
    pointcloud = np.array(pointcloud)
    if intensity_channel:
        return np.array(full_pointcloud.points)
    else:
        return pointcloud

#Guardar pointcloud como .ply
def save_ply(pointcloud, save_path):
    points = pointcloud[:, 0:3]
    points = pd.DataFrame(points)
    points.columns = ['x', 'y', 'z']
    pcd = PyntCloud(points)
    pcd.to_file(save_path)
    #points = pointcloud[:, 0:3]
    #pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    #out = o3d.io.write_point_cloud(save_path, pcd)

#Guardar pointcloud como .bin
def save_bin(pointcloud, save_path):
    pointcloud.astype('float32').tofile(save_path)

#Función para visualizar nube de puntos en formato (x,y,z,reflection o distancia)
def viz_lidar(cloudpoints, title, graph='3d', reduced_pointcloud=True, pointsize=0.2, xlim3d=(-50,80), ylim3d=(-10,20), zlim3d=(-3,8), figsize=(20,10)):

    #graph:punto de vista de la nube de puntos
    #reduced_pointcloud: True--> toma menos puntos para acelerar la visualización

    if reduced_pointcloud == True:
        points_step = int(1. / pointsize)
        point_size = 0.01 * (1. / pointsize)
        velo_range = range(0, cloudpoints.shape[0], points_step)
        velo_frame = cloudpoints[velo_range, :]  

        x = velo_frame[:,0]
        y = velo_frame[:,1]
        z = velo_frame[:,2]
        refl = velo_frame[:,3]

    else:
        x = cloudpoints[:,0]
        y = cloudpoints[:,1]
        z = cloudpoints[:,2]
        refl = cloudpoints[:,3]

    if graph == '3d':
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
        ax.scatter3D(x,y,z, s=pointsize, c=refl, cmap='gray')
        ax.set_xlim3d(xlim3d)
        ax.set_ylim3d(ylim3d)
        ax.set_zlim3d(zlim3d)
        ax.set_title(f'3D pointcloud + {title}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
    elif graph == 'xy':
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.scatter(x,y,s=pointsize, c=refl, cmap='gray')
        ax.set_xlim(xlim3d)
        ax.set_ylim(ylim3d)
        ax.grid(True)
        ax.set_title(f'XY pointcloud + {title}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
    elif graph == 'xz':
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.scatter(x,z,s=pointsize, c=refl, cmap='gray')
        ax.set_xlim(xlim3d)
        ax.set_ylim(zlim3d)
        ax.grid(True)
        ax.set_title(f'XZ pointcloud + {title}')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Z axis')
    elif graph == 'yz':
        fig, ax = plt.subplots(1,1, figsize=figsize)
        ax.scatter(y,z,s=pointsize, c=refl, cmap='gray')
        ax.set_xlim(ylim3d)
        ax.set_ylim(zlim3d)
        ax.grid(True)
        ax.set_title(f'YZ pointcloud + {title}')
        ax.set_xlabel('Y axis')
        ax.set_ylabel('Z axis')
    
#Función para visualizar dos nubes de puntos en formato (x,y,z,reflection o distancia)
def viz_lidar_2(cloudpoint_1, cloudpoint_2, labels, graph='3d', reduced_pointcloud=True, pointsize=0.3, xlim3d=(-50,80), ylim3d=(-10,20), zlim3d=(-3,8), figsize=(20,10)):

  #graph:punto de vista de la nube de puntos
  #reduced_pointcloud: True--> toma menos puntos para acelerar la visualización

  if reduced_pointcloud == True:
    points_step = int(1. / pointsize)
    point_size = 0.01 * (1. / pointsize)
    velo_range_1 = range(0, cloudpoint_1.shape[0], points_step)
    velo_frame_1 = cloudpoint_1[velo_range_1, :]
    velo_range_2 = range(0, cloudpoint_2.shape[0], points_step)
    velo_frame_2 = cloudpoint_2[velo_range_2, :]    

    x_1 = velo_frame_1[:,0]
    y_1 = velo_frame_1[:,1]
    z_1 = velo_frame_1[:,2]
    refl_1 = velo_frame_1[:,3]

    x_2 = velo_frame_2[:,0]
    y_2 = velo_frame_2[:,1]
    z_2 = velo_frame_2[:,2]
    refl_2 = velo_frame_2[:,3]
  
  else:
    x_1 = cloudpoint_1[:,0]
    y_1 = cloudpoint_1[:,1]
    z_1 = cloudpoint_1[:,2]
    refl_1 = cloudpoint_1[:,3]

    x_2 = cloudpoint_2[:,0]
    y_2 = cloudpoint_2[:,1]
    z_2 = cloudpoint_2[:,2]
    refl_2 = cloudpoint_2[:,3]
  
  if graph == '3d':
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(x_1,y_1,z_1, s=pointsize, c=refl_1, cmap='autumn', label= labels[0])
    ax.scatter3D(x_2,y_2,z_2, s=pointsize, c=refl_2, cmap='winter', label = labels[1])
    ax.set_xlim3d(xlim3d)
    ax.set_ylim3d(ylim3d)
    ax.set_zlim3d(zlim3d)
    ax.set_title('3D pointcloud')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.legend(loc="upper right", fontsize=14)
  elif graph == 'xy':
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.scatter(x_1,y_1,s=pointsize, c=refl_1, cmap='autumn', label= labels[0])
    ax.scatter(x_2,y_2,s=pointsize, c=refl_2, cmap='winter', label = labels[1])
    ax.set_xlim(xlim3d)
    ax.set_ylim(ylim3d)
    ax.grid(True)
    ax.set_title('XY pointcloud')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    plt.legend(loc="upper right", fontsize=14)
  elif graph == 'xz':
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.scatter(x_1,z_1,s=pointsize, c=refl_1, cmap='autumn', label= labels[0])
    ax.scatter(x_2,z_2,s=pointsize, c=refl_2, cmap='winter', label = labels[1])
    ax.set_xlim(xlim3d)
    ax.set_ylim(zlim3d)
    ax.grid(True)
    ax.set_title('XZ pointcloud')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Z axis')
    plt.legend(loc="upper right", fontsize=14)
  elif graph == 'yz':
    fig, ax = plt.subplots(1,1, figsize=figsize)
    ax.scatter(y_1,z_1,s=pointsize, c=refl_1, cmap='autumn', label= labels[0])
    ax.scatter(y_2,z_2,s=pointsize, c=refl_2, cmap='winter', label = labels[1])
    ax.set_xlim(ylim3d)
    ax.set_ylim(zlim3d)
    ax.grid(True)
    ax.set_title('YZ pointcloud')
    ax.set_xlabel('Y axis')
    ax.set_ylabel('Z axis')
    plt.legend(loc="upper right", fontsize=14)

#Función para visualizar una imagen de rango
def display_range_image(img, h_res=0.35, v_fov=(-24.9, 2.0), h_fov=(-180, 180), cmap='jet'):
    '''
    img: range image
    h_res: horizontal resolution
    v_fov: vertical field of view (ej: (-24.9,2.0))
    h_fov: horizontal field of view (ej: (-180,180))
    '''
    def scale_xaxis(axis_value, *args):
        if h_fov[0] > 0:
            return int(np.round((axis_value * h_res + h_fov[0])))
        else:
            return int(np.round((axis_value * h_res - abs(h_fov[0]))))

    plt.subplots(1,1, figsize = (30,20))
    plt.title(f"Range Image - Vertical FOV ({v_fov[0]}º, {v_fov[1]}º) & Horizontal FOV ({h_fov[0]}º , {h_fov[1]}º) - 0º means the front of the car")
    plt.imshow(img, cmap=cmap)
    plt.xticks(np.arange(0,len(img[1]),len(img[1])/8))
    formatter = FuncFormatter(scale_xaxis)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel("Rotate angle [º]")
    plt.ylabel("Channels")
    plt.show()

    print(f"Size: {img.shape}")


def read_labels(labels_path):
    labels = []
    for label_path in labels_path:
        label_data = pd.read_csv(label_path, sep=' ', header=None, names=['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
                                                                            'height_object', 'width_object', 'length_object', 'location_x_camera', 'location_y_camera', 
                                                                            'location_z_camera', 'rotation_y'])
        label_data = label_data[label_data['type']!='DontCare']
        #label_data = label_data.drop(columns=['type'])
        labels.append(label_data.values.tolist())
        #label_data = np.array([label_data])
    return labels