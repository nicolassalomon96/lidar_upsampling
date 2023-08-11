#Generación del dataset con generadores agregando transformaciones (ruido, flip vertical y horizontal)
import numpy as np
import os
import cv2
import imageio.v2 as io
from config import *

intensity_images_name = os.listdir(lr_intensity_folder) #lr images and hr images have the same name
intensity_test_images_name = os.listdir(lr_intensity_test_folder)

intensity_train_n = round(len(intensity_images_name)*train_images_percent) #Use 90% of the dataset for training

#Random shuffle of the dataset
intensity_rand_images = np.copy(intensity_images_name)
np.random.shuffle(intensity_rand_images)

#Train/Valid
intensity_train_urls = intensity_rand_images[:intensity_train_n]
intensity_valid_urls = intensity_rand_images[intensity_train_n:]

print(f'Total intensity images: {len(intensity_images_name)}')
print(f'Train intensity images: {len(intensity_train_urls)}')
print(f'Validation intensity images: {len(intensity_valid_urls)}')
print(f'Test intensity images: {len(intensity_test_images_name)}')

def data_augmentation(lrimg,hrimg,dataset=None,mode='Train'):

    #No es necesario normalizar ya que en intensidad, los pixeles ya estan en el rango 0 - 1.0
    '''if dataset == 'kitti':
        lrimg = np.array(lrimg * (1./kitti_max_distance), dtype=np.float32)
        hrimg = np.array(hrimg * (1./kitti_max_distance), dtype=np.float32)

    if dataset == 'carla':
        lrimg = np.array(lrimg * (1./carla_max_distance), dtype=np.float32)
        hrimg = np.array(hrimg * (1./carla_max_distance), dtype=np.float32)'''


    if mode =='Train' or mode == 'Valid':
        #Horizontal flip
        if np.random.uniform() > 0.5:
            lrimg = cv2.flip(lrimg,1)
            hrimg = cv2.flip(hrimg,1)
        
        #Vertical flip
        if np.random.uniform() > 0.5:
            lrimg = cv2.flip(lrimg,0)
            hrimg = cv2.flip(hrimg,0)

        if dataset == 'carla':
            #Add noise
            if np.random.uniform() > 0.5:
                noise = np.random.normal(0, noise_std, lrimg.shape) # mu, sigma, size
                lrimg = lrimg + noise

    lrimg = np.expand_dims(lrimg, axis=2)
    hrimg = np.expand_dims(hrimg, axis=2)
    return lrimg, hrimg

def train_data_generator():
    mode ='Train'
    while True:
        for _, url in enumerate(intensity_train_urls):
            if url[0:5] == 'drive':
                dataset = 'kitti'
            if url[0:4] == 'Town':
                dataset = 'carla'
            #Get random images
            idx = np.random.randint(0,len(intensity_train_urls))
            lrimg = io.imread(lr_intensity_folder + '\\' + intensity_train_urls[idx])
            hrimg = io.imread(hr_intensity_folder + '\\' + intensity_train_urls[idx])
            
            lrimg, hrimg = data_augmentation(lrimg, hrimg, dataset, mode)
            yield (lrimg, hrimg)

def valid_data_generator():
    mode ='Valid'
    while True:
        for _,url in enumerate(intensity_valid_urls):
            if url[0:5] == 'drive':
                dataset = 'kitti'
            if url[0:4] == 'Town':
                dataset = 'carla'
            #Get random images
            idx = np.random.randint(0,len(intensity_valid_urls))
            lrimg = io.imread(lr_intensity_folder + '\\' + intensity_valid_urls[idx])
            hrimg = io.imread(hr_intensity_folder + '\\' + intensity_valid_urls[idx])
        
            lrimg, hrimg = data_augmentation(lrimg, hrimg, dataset, mode)
            yield (lrimg, hrimg)

# No se usa porque ya separé manualemnte imágenes para test
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