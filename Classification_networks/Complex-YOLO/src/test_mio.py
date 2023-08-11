import argparse
import sys
import os
import time
import imageio.v2 as io

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    parser.add_argument('--saved_fn', type=str, default='complexer_yolov4', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='the threshold for conf')

    parser.add_argument('--show_image', action='store_true',
                        help='If true, show the image during demostration')
    parser.add_argument('--save_test_output', action='store_true',
                        help='If true, the output image of the testing phase will be saved')
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output_video_fn', type=str, default='out_complexer_yolov4', metavar='PATH',
                        help='the video filename if the output format is video')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

     ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    configs.dataset_dir = os.path.join(configs.working_dir, 'dataset', 'kitti')

    if configs.save_test_output:
        configs.results_dir = os.path.join(configs.working_dir, 'results', configs.saved_fn)
        make_folder(configs.results_dir)

    return configs

#Leer range image y devolver pointcloud (formato: x,y,z,distancia o intensidad (dependiendo del tipo de imagen de rango))
def range_image_to_pointcloud(image, v_fov=(-24.9, 2.0), h_fov=(-180, 180)):
    '''
    image: range image en formato 0 - max_distance
    v_fov: vertical field of view (ej: (-24.9,2.0))
    h_fov: horizontal field of view (ej: (-180,180))
    '''
    kitti_carla_min_range = 3.0
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

    return pointcloud[0]

if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False  # For testing

    model = create_model(configs)
    model.print_network()
    print('\n\n' + '-*=' * 30 + '\n\n')
    
    device_string = 'cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx)
    
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)
    model.load_state_dict(torch.load(configs.pretrained_path, map_location=device_string))

    configs.device = torch.device(device_string)
    model = model.to(device=configs.device)

    out_cap = None

    model.eval()

    print("MODELO CREADO")

    # Obtener la BEV RGB de la nube de puntos

    real_image = io.imread(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\unet\test_01_hr.tif')
    #low_image = io.imread(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\unet\test_01_lr.tif')
    low_image = io.imread(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\Datasets LIDAR\kiiti_carla_merged\range_image_64ch_test\drive_0019_0000000406.tif')
    generated_image = io.imread(r'D:\Nicolas\Posgrado\Trabajos y Tesis\LIDAR\LIDAR_super_resolution\Scripts\unet\test_01_pred.tif')

    pointcloud_64ch = range_image_to_pointcloud(real_image, (-24.9,2.0), (-180,180))
    pointcloud_64ch = pointcloud_64ch[0]
    #print(pointcloud_64ch.shape)

    pointcloud_16ch = range_image_to_pointcloud(low_image, (-24.9,2.0), (-180,180))
    pointcloud_16ch = pointcloud_16ch[0]
    #print(pointcloud_16ch.shape)

    pointcloud_gen = range_image_to_pointcloud(generated_image, (-24.9,2.0), (-180,180))
    pointcloud_gen = pointcloud_gen[0]
    #print(pointcloud_gen.shape)

    b_16ch = kitti_bev_utils.removePoints(pointcloud_16ch, cnf.boundary)
    rgb_map_16ch = kitti_bev_utils.makeBVFeature(b_16ch, cnf.DISCRETIZATION, cnf.boundary)
    #rgb_16ch = np.dstack([rgb_map_16ch[0,:,:], rgb_map_16ch[1,:,:], rgb_map_16ch[2,:,:]])
    rgb_16ch = np.moveaxis(rgb_map_16ch, 0, 2)

    b_64ch = kitti_bev_utils.removePoints(pointcloud_64ch, cnf.boundary)
    rgb_map_64ch = kitti_bev_utils.makeBVFeature(b_64ch, cnf.DISCRETIZATION, cnf.boundary)
    #rgb_64ch = np.dstack([rgb_map_64ch[1,:,:], rgb_map_64ch[2,:,:], rgb_map_64ch[0,:,:]])
    rgb_64ch = np.moveaxis(rgb_map_64ch, 0, 2)

    b_64chgen = kitti_bev_utils.removePoints(pointcloud_gen, cnf.boundary)
    rgb_map_64chgen = kitti_bev_utils.makeBVFeature(b_64chgen, cnf.DISCRETIZATION, cnf.boundary)
    #rgb_64chgen = np.dstack([rgb_map_64chgen[1,:,:], rgb_map_64chgen[2,:,:], rgb_map_64chgen[0,:,:]])
    rgb_64chgen = np.moveaxis(rgb_map_64chgen, 0, 2)
  

    cv2.imshow('Imagen 16 canales - Presione 0 para salir', rgb_16ch)
    cv2.imshow('Imagen 64 canales - Presione 0 para salir', rgb_64ch)
    cv2.imshow('Imagen 64 canales generados - Presione 0 para salir', rgb_64chgen)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    
    imgs_bev = torch.from_numpy(np.array([rgb_map_16ch]))

    with torch.no_grad():
        input_imgs = imgs_bev.to(device=configs.device).float()
        t1 = time_synchronized()
        outputs = model(input_imgs)
        t2 = time_synchronized()
        detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)

        img_bev = imgs_bev.squeeze() * 255
        img_bev = img_bev.permute(1, 2, 0).numpy().astype(np.uint8)
        img_bev = cv2.resize(img_bev, (configs.img_size, configs.img_size))
        for detections in img_detections:
            if detections is None:
                continue
            # Rescale boxes to original image
            detections = rescale_boxes(detections, configs.img_size, img_bev.shape[:2])
            for x, y, w, l, im, re, *_, cls_pred in detections:
                yaw = np.arctan2(im, re)
                # Draw rotated box
                kitti_bev_utils.drawRotatedBox(img_bev, x, y, w, l, yaw, cnf.colors[int(cls_pred)])
        
        cv2.imshow('Imagen - Presione 0 para salir', img_bev)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
                
        #img_bev = cv2.flip(cv2.flip(img_bev, 0), 1)

        out_img = img_bev
    
    print(img_detections)
    #io.imsave(r'D:\Nicolas\Posgrado\Presentación avances\Lidar_Super_Resolution_Octubre_2022\resultado.jpg', out_img)