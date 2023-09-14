a = 1
import numpy as np
import pandas as pd
import os
import cv2
import imageio.v2 as io
from config import *

velodyne_name = os.listdir(velodyne_folder) #lr images and hr images have the same name
labels_name = os.listdir(labels_folder) #folder with all labels

train_n = round(len(velodyne_name)*train_images_percent) # Train-valid split

#Random shuffle of the dataset
rand_velo = np.copy(velodyne_name)
np.random.shuffle(rand_velo)

#Train/Valid
train_urls = rand_velo[:train_n]
valid_urls = rand_velo[train_n:]

print(f'Total velodyne images: {len(velodyne_name)}')
print(f'Train velodyne images: {len(train_urls)}')
print(f'Validation velodyne images: {len(valid_urls)}')

def data_augmentation(lrimg, hrimg, dataset=None, augment=True):

    def add_random_boxes(img, n_boxes, box_size=(3,3)):
        h,w = box_size[0], box_size[1]
        for _ in range(n_boxes):
            y_box = np.random.randint(0, img.shape[0] - h)
            x_box = np.random.randint(0, img.shape[1] - w)
            img[y_box:y_box+h, x_box:x_box+w] = 0.0
        return img

    #Replace all sub-zero and upper max values because it is impossible in range images
    lrimg[lrimg < kitti_carla_min_range] = 0.0
    hrimg[hrimg < kitti_carla_min_range] = 0.0

    if dataset == 'kitti':
        lrimg[lrimg > kitti_max_distance] = 0.0
        hrimg[hrimg > kitti_max_distance] = 0.0

        lrimg = np.array(lrimg * (1./kitti_max_distance), dtype=np.float32)
        hrimg = np.array(hrimg * (1./kitti_max_distance), dtype=np.float32)
    
    elif dataset == 'carla':
        lrimg[lrimg > carla_max_distance] = 0.0
        hrimg[hrimg > carla_max_distance] = 0.0

        lrimg = np.array(lrimg * (1./carla_max_distance), dtype=np.float32)
        hrimg = np.array(hrimg * (1./carla_max_distance), dtype=np.float32)

    #Standarize data
    #lrimg = np.array((lrimg - np.mean(lrimg)) / (np.std(lrimg)), dtype=np.float32)
    #hrimg = np.array((hrimg - np.mean(hrimg)) / (np.std(hrimg)), dtype=np.float32)

    if augment:
        
        #Horizontal flip: it is not used because if you flip the image, the labels stop matching and Chamfer_loss is Nan
        #if np.random.uniform() > 0.5:
        #    lrimg = cv2.flip(lrimg,1)
        #    hrimg = cv2.flip(hrimg,1)
        
        #Vertical flip
        #if np.random.uniform() > 0.5:
        #    lrimg = cv2.flip(lrimg,0)
        #    hrimg = cv2.flip(hrimg,0)

        #Add noise
        if np.random.uniform() > 0.5:
            noise_lr_std = lrimg.std() * 0.1
            noise = np.random.normal(0, noise_lr_std, lrimg.shape) # mu, sigma, size
            lrimg = (lrimg + noise).astype(np.float32)
            
        #Add black boxes
        if np.random.uniform() > 0.5:
            lrimg = add_random_boxes(lrimg, n_boxes=100, box_size=(5,10))

    lrimg = np.expand_dims(lrimg, axis=0)
    hrimg = np.expand_dims(hrimg, axis=0)
    return lrimg, hrimg


def train_data_generator():
    augment = True
    np.random.shuffle(train_urls)

    for _, url in enumerate(train_urls):
        if url[0:5] == 'drive':
            dataset = 'kitti'
        if url[0:4] == 'Town':
            dataset = 'carla'

        #Get random images
        train_path = os.path.join(velodyne_folder, url) #hr_folder + '\\' + url

        if train_path[-3:] == 'npy':
            hrimg = np.load(train_path)
        elif train_path[-3:] == 'tif':
            hrimg = io.imread(train_path)
        else:
            print("Wrong pointcloud filepath")
        
        indexes = range(0, high_res_height, upsampling_ratio)
        lrimg = hrimg[indexes]
        lrimg, hrimg = data_augmentation(lrimg, hrimg, dataset, augment)

        #Get labels for each image
        label_url = url.split(sep='.')[0] + '.txt' #drive_001922.txt
        label_url = label_url.split(sep='_')[1] #001922.txt
        train_labels_path = os.path.join(labels_folder, label_url)

        yield (lrimg, hrimg, train_labels_path)


def valid_data_generator():
    augment = True
    np.random.shuffle(valid_urls)

    for _, url in enumerate(valid_urls):
        if url[0:5] == 'drive':
            dataset = 'kitti'
        if url[0:4] == 'Town':
            dataset = 'carla'
        
        #Get random images
        valid_path = os.path.join(velodyne_folder, url) #hr_folder + '\\' + url
        if valid_path[-3:] == 'npy':
            hrimg = np.load(valid_path)
        elif valid_path[-3:] == 'tif':
            hrimg = io.imread(valid_path)
        else:
            print("Wrong pointcloud filepath")
    
        indexes = range(0, high_res_height, upsampling_ratio)
        lrimg = hrimg[indexes]
        lrimg, hrimg = data_augmentation(lrimg, hrimg, dataset, augment)

        #Get labels for each image
        label_url = url.split(sep='.')[0] + '.txt' #drive_001922.txt
        label_url = label_url.split(sep='_')[1] #001922.txt
        valid_labels_path = os.path.join(labels_folder, label_url)

        yield (lrimg, hrimg, valid_labels_path)