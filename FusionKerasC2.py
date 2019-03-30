import tensorflow as tf
import cv2
from skimage import io
from matplotlib import pyplot as plt

# from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import load_model
import numpy as np
import dataset
import os

os.environ['CUDA_VISIBLE_DEVICES']='4'

image_size = 48
num_channels = 2
# images = []

path1 = "/media/data2/mhy/LytroDataset/lytro-00-A.jpg"
path2 = "/media/data2/mhy/LytroDataset/lytro-00-B.jpg"

img1=io.imread(path1,as_gray=True)
# io.imshow(img1)
# plt.show()
img2=io.imread(path2,as_gray=True)
# io.imshow(img2)
# plt.show()

io.imsave('./ImgData/F1.jpg',img1)
io.imsave('./ImgData/F2.jpg',img1)

img1 = np.array(img1, dtype=np.uint8)
img1 = img1.astype('float32')
img1 = np.multiply(img1, 1.0 / 255.0)
img2 = np.array(img2, dtype=np.uint8)
img2 = img2.astype('float32')
img2 = np.multiply(img2, 1.0 / 255.0)

FusionResult=io.imread('./ImgData/F1.jpg',as_gray=True)
FusionResult = FusionResult.astype('float32')
FusionResult = np.multiply(FusionResult, 1.0 / 255.0)
FusionMap=io.imread('./ImgData/F2.jpg',as_gray=True)
FusionMap = FusionMap.astype('float32')
FusionMap = np.multiply(FusionMap, 1.0 / 255.0)

model = load_model('/media/data2/mhy/data/0927N2/Models2/ResNet_ResNet56v2_model.007.h5')

image_r=int(image_size/2)
for i in range(image_r,img1.shape[0]-image_r):
    for j in range(image_r,img1.shape[1]-image_r):
        patch1=img1[i-image_r:i+image_r,j-image_r:j+image_r]
        patch2=img2[i-image_r:i+image_r,j-image_r:j+image_r]

        patch=np.zeros([image_size,image_size,2])
        patch[:,:,0]=patch2
        patch[:,:,1]=patch1

        x_patch = patch.reshape(1, image_size, image_size, num_channels)

        # 获取默认的图
        preds=model.predict(x_patch)

        if preds[0,0]>preds[0,1]:
            FusionMap[i, j] = 1
            FusionResult[i, j] = img1[i, j]
        else:
            FusionMap[i, j] = 0
            FusionResult[i, j] = img2[i, j]


        # print(result1[0, 0])
        # print(result1[0, 1])
        #
        # if result2[0,1]>result1[0,1]:
        #     FusionResult[i,j]=img2[i,j]
        #     FusionMap[i, j]=0
        # else:
        #     FusionMap[i, j] = 1


io.imshow(FusionResult)
plt.show()
io.imshow(FusionMap)
plt.show()

io.imsave('./ImgData/FusionResult.png',FusionResult)
io.imsave('./ImgData/FusionMap.png',FusionMap)


