
from skimage import io
from matplotlib import pyplot as plt
from keras.models import load_model
import scipy as sc
import numpy as np
from PIL import Image
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES']='7'

image_size = 32
num_channels = 4
# images = []

image_r=int(image_size/2)
model = load_model('/media/data2/mhy/data/1015N1/ModelsN/ResNet_ResNet56v2_model.017.h5')

for ii in range(24,25):

    path1 = "/media/data2/mhy/LytroDataset/SC/lytro-%02d-A.jpg"%(ii)
    path2 = "/media/data2/mhy/LytroDataset/SC/lytro-%02d-B.jpg"%(ii)
    print(path1)
    print(path2)

    img1=io.imread(path1,as_gray=True)
    # io.imshow(img1)
    # plt.show()
    img2=io.imread(path2,as_gray=True)
    # io.imshow(img2)
    # plt.show()

    io.imsave("/media/data2/mhy/data/1015N1/results_Res56v2_p322_e17/L%02d-A-Source.jpg"%(ii),img1)
    io.imsave("/media/data2/mhy/data/1015N1/results_Res56v2_p322_e17/L%02d-B-Source.jpg"%(ii),img2)

    # img1 = img1.astype('float32')
    # img1 = np.multiply(img1, 1.0 / 255.0)
    # img2 = img2.astype('float32')
    # img2 = np.multiply(img2, 1.0 / 255.0)

    ELimg1 = cv2.copyMakeBorder(img1, image_r * 2, image_r * 2, image_r * 2, image_r * 2, cv2.BORDER_REFLECT)
    ELimg2 = cv2.copyMakeBorder(img2, image_r * 2, image_r * 2, image_r * 2, image_r * 2, cv2.BORDER_REFLECT)

    FusionResult = io.imread('/media/data2/mhy/data/1015N1/results_Res56v2_p322_e17/L%02d-A-Source.jpg'%(ii), as_gray=True)
    FusionResult = FusionResult.astype('float32')
    FusionResult = np.multiply(FusionResult, 1.0 / 255.0)
    ScoreMap = io.imread('/media/data2/mhy/data/1015N1/results_Res56v2_p322_e17/L%02d-A-Source.jpg' % (ii), as_gray=True)
    ScoreMap = ScoreMap.astype('float32')
    ScoreMap = np.multiply(ScoreMap, 1.0 / 255.0)
    FusionMap = io.imread('/media/data2/mhy/data/1015N1/results_Res56v2_p322_e17/L%02d-A-Source.jpg'%(ii), as_gray=True)
    FusionMap = FusionMap.astype('float32')
    FusionMap = np.multiply(FusionMap, 1.0 / 255.0)

    for i in range(image_r*2, img1.shape[0] + image_r*2):
        for j in range(image_r*2, img1.shape[1] + image_r*2):
            patch1 = ELimg1[i - image_r*2:i + image_r*2, j - image_r*2:j + image_r*2]
            patch2 = ELimg2[i - image_r*2:i + image_r*2, j - image_r*2:j + image_r*2]

            patch11 = ELimg1[i - image_r:i + image_r, j - image_r:j + image_r]
            patch21 = ELimg2[i - image_r:i + image_r, j - image_r:j + image_r]

            patch12 = cv2.resize(patch1, (image_size, image_size))
            patch22 = cv2.resize(patch2, (image_size, image_size))

            patch = np.zeros([image_size, image_size, 4])

            patch[:, :, 0] = patch11
            patch[:, :, 1] = patch21
            patch[:, :, 2] = patch12
            patch[:, :, 3] = patch22

            # patch[:, :, 0] = patch22
            # patch[:, :, 1] = patch12

            x_patch = patch.reshape(1, image_size, image_size, num_channels)

            # 获取默认的图
            preds = model.predict(x_patch)
            ScoreMap[i - image_r*2, j - image_r*2] = preds[0, 0]

            if preds[0, 0] > preds[0, 1]:
                FusionMap[i-image_r*2, j-image_r*2] = 1
                FusionResult[i-image_r*2, j-image_r*2] = img1[i-image_r*2, j-image_r*2]
            else:
                FusionMap[i-image_r*2, j-image_r*2] = 0
                FusionResult[i-image_r*2, j-image_r*2] = img2[i-image_r*2, j-image_r*2]







    io.imsave('/media/data2/mhy/data/1015N1/results_Res56v2_p322_e17/L%02d-Result.png' % (ii), FusionResult)
    io.imsave('/media/data2/mhy/data/1015N1/results_Res56v2_p322_e17/L%02d-ScoreMap.png' % (ii), ScoreMap)
    io.imsave('/media/data2/mhy/data/1015N1/results_Res56v2_p322_e17/L%02d-FusionMap.png' % (ii), FusionMap)

    # io.imshow(FusionResult)
    # plt.show()
    # io.imshow(FusionMap)
    # plt.show()

