import numpy as np
import os
import cv2
import imageio.v2 as io
from config import *

images_name = os.listdir(hr_folder) #lr images and hr images have the same name
train_n = round(len(images_name)*train_images_percent) # Train-valid split

distance_3d_images = os.listdir(hr_folder)
intensity_3d_images = os.listdir(hr_intensity_folder)

#Random shuffle of the dataset
rand_images = np.copy(images_name)
np.random.shuffle(rand_images)

#Train/Valid
train_urls = rand_images[:train_n]
valid_urls = rand_images[train_n:]

print(f'Total distance images: {len(images_name)}')
print(f'Train distance images: {len(train_urls)}')
print(f'Validation distance images: {len(valid_urls)}')


def data_augmentation(lrimg, hrimg, dataset=None, augment=True):

    def add_random_boxes(img, n_boxes, box_size=(3,3)):
        h,w = box_size[0], box_size[1]
        for _ in range(n_boxes):
            y_box = np.random.randint(0, img.shape[0] - h)
            x_box = np.random.randint(0, img.shape[1] - w)
            img[y_box:y_box+h, x_box:x_box+w] = 0.0
        return img

    #Replace all sub-zero and upper max values because it is impossible in range images
    #lrimg[lrimg < kitti_carla_min_range] = 0.0
    #hrimg[hrimg < kitti_carla_min_range] = 0.0

    distance_lrimg = lrimg[:,:,:1]
    intensity_lrimg = lrimg[:,:,1:] 
    distance_hrimg = hrimg[:,:,:1]
    intensity_hrimg = hrimg[:,:,1:]

    distance_lrimg[distance_lrimg < kitti_carla_min_range] = 0.0
    distance_hrimg[distance_hrimg < kitti_carla_min_range] = 0.0

    if dataset == 'kitti':
        distance_lrimg[distance_lrimg > kitti_max_distance] = 0.0
        distance_hrimg[distance_hrimg > kitti_max_distance] = 0.0
    elif dataset == 'carla':
        distance_lrimg[distance_lrimg > carla_max_distance] = 0.0
        distance_hrimg[distance_hrimg > carla_max_distance] = 0.0

    #Normalization
    if dataset == 'kitti':
        distance_lrimg = np.array(distance_lrimg * (1./kitti_max_distance), dtype=np.float32)
        distance_hrimg = np.array(distance_hrimg * (1./kitti_max_distance), dtype=np.float32)
    
    if dataset == 'carla':
        distance_lrimg = np.array(distance_lrimg * (1./carla_max_distance), dtype=np.float32)
        distance_hrimg = np.array(distance_hrimg * (1./carla_max_distance), dtype=np.float32)
    
    #Normalize data to [0-1]
    #lrimg = np.array((lrimg - np.min(lrimg)) / (np.max(lrimg) - np.min(lrimg)), dtype=np.float32)
    #hrimg = np.array((hrimg - np.min(hrimg)) / (np.max(hrimg) - np.min(hrimg)), dtype=np.float32)

    #Standarize data
    #lrimg = np.array((lrimg - np.mean(lrimg)) / (np.std(lrimg)), dtype=np.float32)
    #hrimg = np.array((hrimg - np.mean(hrimg)) / (np.std(hrimg)), dtype=np.float32)

    #print(lrimg.shape)

    lrimg = np.dstack([distance_lrimg, intensity_lrimg])
    hrimg = np.dstack([distance_hrimg, intensity_hrimg])

    if augment:
        
        #Horizontal flip
        if np.random.uniform() > 0.5:
            lrimg = cv2.flip(lrimg,1)
            hrimg = cv2.flip(hrimg,1)
        
        #Vertical flip
        #if np.random.uniform() > 0.5:
        #    lrimg = cv2.flip(lrimg,0)
        #    hrimg = cv2.flip(hrimg,0)

        #Add noise
        #if np.random.uniform() > 0.5:
        #    noise_lr_std = lrimg.std() * 0.1
        #    noise = np.random.normal(0, noise_lr_std, lrimg.shape) # mu, sigma, size
        #    lrimg = (lrimg + noise).astype(np.float32)
            
        #Add black boxes
        #if np.random.uniform() > 0.5:
        #    lrimg = add_random_boxes(lrimg, n_boxes=100, box_size=(5,10))

    lrimg = np.moveaxis(lrimg, -1, 0)
    hrimg = np.moveaxis(hrimg, -1, 0)
    #lrimg = np.expand_dims(lrimg, axis=0)
    #hrimg = np.expand_dims(hrimg, axis=0)
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
        train_path = os.path.join(hr_folder, url) #hr_folder + '\\' + url

        if train_path[-3:] == 'npy':
            hrimg = np.load(train_path)
        elif train_path[-3:] == 'tif':
            hrimg = io.imread(train_path)
        else:
            print("Wrong pointcloud filepath")
        
        indexes = range(0, high_res_height, upsampling_ratio)
        lrimg = hrimg[indexes]
        lrimg, hrimg = data_augmentation(lrimg, hrimg, dataset, augment)

        yield (lrimg, hrimg)


def valid_data_generator():
    augment = True
    np.random.shuffle(valid_urls)

    for _, url in enumerate(valid_urls):
        if url[0:5] == 'drive':
            dataset = 'kitti'
        if url[0:4] == 'Town':
            dataset = 'carla'
        
        #Get random images
        valid_path = os.path.join(hr_folder, url) #hr_folder + '\\' + url
        if valid_path[-3:] == 'npy':
            hrimg = np.load(valid_path)
        elif valid_path[-3:] == 'tif':
            hrimg = io.imread(valid_path)
        else:
            print("Wrong pointcloud filepath")
    
        indexes = range(0, high_res_height, upsampling_ratio)
        lrimg = hrimg[indexes]
        lrimg, hrimg = data_augmentation(lrimg, hrimg, dataset, augment)

        yield (lrimg, hrimg)

def test_data_generator():
    augment = False

    for _, url in enumerate(distance_3d_images):
        if url[0:5] == 'drive':
            dataset = 'kitti'
        if url[0:4] == 'Town':
            dataset = 'carla'
        
        #Get images
        distance_images_path = os.path.join(hr_folder, url) #hr_folder + '\\' + url
        intensity_images_path = os.path.join(hr_intensity_folder, url) #hr_folder + '\\' + url
        
        if distance_images_path[-3:] == 'npy':
            hrimg_distance = np.load(distance_images_path)
            hrimg_intensity = np.load(intensity_images_path)
        elif distance_images_path[-3:] == 'tif':
            hrimg_distance = io.imread(distance_images_path)
            hrimg_intensity = io.imread(intensity_images_path)
        else:
            print("Wrong pointcloud filepath")
        
        hrimg = np.dstack([hrimg_distance, hrimg_intensity])
    
        indexes = range(0, high_res_height, upsampling_ratio)
        lrimg = hrimg[indexes]
        lrimg, hrimg = data_augmentation(lrimg, hrimg, dataset, augment)

        yield (lrimg, hrimg)