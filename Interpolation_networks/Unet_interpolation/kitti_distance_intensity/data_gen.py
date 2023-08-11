#Generación del dataset con generadores agregando transformaciones (ruido, flip vertical y horizontal)
import numpy as np
import os
import cv2
import imageio.v2 as io
from config import *

images_name = os.listdir(lr_folder) #lr images and hr images have the same name
#test_images_name = os.listdir(lr_test_folder)

train_n = round(len(images_name)*train_images_percent) #Use 90% of the dataset for training

#Random shuffle of the dataset
rand_images = np.copy(images_name)
np.random.shuffle(rand_images)

#Train/Valid
train_urls = rand_images[:train_n]
valid_urls = rand_images[train_n:]

print(f'Total distance images: {len(images_name)}')
print(f'Train distance images: {len(train_urls)}')
print(f'Validation distance images: {len(valid_urls)}')
#print(f'Test distance images: {len(test_images_name)}')

def data_augmentation(lrimg,hrimg,dataset=None,mode='Train'):

    #First thing to do --> Normalization
    if dataset == 'kitti':
        lrimg = np.array(lrimg * (1./kitti_max_distance), dtype=np.float32)
        hrimg = np.array(hrimg * (1./kitti_max_distance), dtype=np.float32)

    if dataset == 'carla':
        lrimg = np.array(lrimg * (1./carla_max_distance), dtype=np.float32)
        hrimg = np.array(hrimg * (1./carla_max_distance), dtype=np.float32)


    if mode =='Train' or mode == 'Valid':
        #Horizontal flip
        if np.random.uniform() > 0.5:
            lrimg = cv2.flip(lrimg,1)
            hrimg = cv2.flip(hrimg,1)
        
        #Vertical flip
        if np.random.uniform() > 0.5:
            lrimg = cv2.flip(lrimg,0)
            hrimg = cv2.flip(hrimg,0)

        #Add noise
        if np.random.uniform() > 0.5:
            noise = np.random.normal(0, noise_std, lrimg.shape) # mu, sigma, size
            lrimg = lrimg + noise

    #print(lrimg.shape)

    #lrimg = np.expand_dims(lrimg, axis=2)
    #hrimg = np.expand_dims(hrimg, axis=2)
    return lrimg, hrimg

def train_data_generator():
    mode ='Train'
    while True:
        for _, url in enumerate(train_urls):
            if url[0:5] == 'drive':
                dataset = 'kitti'
            if url[0:4] == 'Town':
                dataset = 'carla'
            #Get random images
            idx = np.random.randint(0,len(train_urls))
            lrimg = np.load(lr_folder + '\\' + train_urls[idx])
            hrimg = io.imread(hr_folder + '\\' + (train_urls[idx]).split(sep='.')[0] + '.tif')
            
            lrimg, hrimg = data_augmentation(lrimg, hrimg, dataset, mode)
            yield (lrimg, hrimg)

def valid_data_generator():
    mode ='Valid'
    while True:
        for _,url in enumerate(valid_urls):
            if url[0:5] == 'drive':
                dataset = 'kitti'
            if url[0:4] == 'Town':
                dataset = 'carla'
            #Get random images
            idx = np.random.randint(0,len(valid_urls))

            lrimg = np.load(lr_folder + '\\' + valid_urls[idx])
            hrimg = io.imread(hr_folder + '\\' + (valid_urls[idx]).split(sep='.')[0] + '.tif')
        
            lrimg, hrimg = data_augmentation(lrimg, hrimg, dataset, mode)
            yield (lrimg, hrimg)

'''      
def test_data_generator():    
    mode = 'Test'
    #while True:
    for _,url in enumerate(test_images_name):
        if url[0:5] == 'drive':
            dataset = 'kitti'
        if url[0:4] == 'Town':
            dataset = 'carla'
        #Get random images
        idx = np.random.randint(0,len(test_images_name))
        lrimg = io.imread(lr_test_folder + '\\' + test_images_name[idx])
        hrimg = io.imread(hr_test_folder + '\\' + test_images_name[idx])

        lrimg, hrimg = data_augmentation(lrimg, hrimg, dataset, mode)
        yield (lrimg, hrimg)'''