
from skimage import io
from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES']='7'

image_size = 64
num_channels = 2
# images = []

image_r=int(image_size/2)
model = load_model('/media/data2/mhy/data/1012N2/Models0/ResNet_ResNet56v2_model.008.h5')

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

    io.imsave("/media/data2/mhy/data/1012N2/results_Res56v2_p64_e08/L%02d-A-Source.jpg"%(ii),img1)
    io.imsave("/media/data2/mhy/data/1012N2/results_Res56v2_p64_e08/L%02d-B-Source.jpg"%(ii),img2)

    # img1 = img1.astype('float32')
    # img1 = np.multiply(img1, 1.0 / 255.0)
    # img2 = img2.astype('float32')
    # img2 = np.multiply(img2, 1.0 / 255.0)

    ELimg1 = cv2.copyMakeBorder(img1, image_r, image_r, image_r, image_r, cv2.BORDER_REFLECT)
    ELimg2 = cv2.copyMakeBorder(img2, image_r, image_r, image_r, image_r, cv2.BORDER_REFLECT)

    FusionResult = io.imread('/media/data2/mhy/data/1012N2/results_Res56v2_p64_e08/L%02d-A-Source.jpg'%(ii), as_gray=True)
    FusionResult = FusionResult.astype('float32')
    FusionResult = np.multiply(FusionResult, 1.0 / 255.0)
    ScoreMap = io.imread('/media/data2/mhy/data/1012N2/results_Res56v2_p64_e08/L%02d-A-Source.jpg' % (ii), as_gray=True)
    ScoreMap = ScoreMap.astype('float32')
    ScoreMap = np.multiply(ScoreMap, 1.0 / 255.0)
    FusionMap = io.imread('/media/data2/mhy/data/1012N2/results_Res56v2_p64_e08/L%02d-A-Source.jpg'%(ii), as_gray=True)
    FusionMap = FusionMap.astype('float32')
    FusionMap = np.multiply(FusionMap, 1.0 / 255.0)

    image_r = int(image_size / 2)
    for i in range(image_r, img1.shape[0] + image_r):
        for j in range(image_r, img1.shape[1] + image_r):
            patch1 = ELimg1[i - image_r:i + image_r, j - image_r:j + image_r]
            patch2 = ELimg2[i - image_r:i + image_r, j - image_r:j + image_r]

            patch = np.zeros([image_size, image_size, 2])
            patch[:, :, 0] = patch2
            patch[:, :, 1] = patch1

            x_patch = patch.reshape(1, image_size, image_size, num_channels)

            # 获取默认的图
            preds = model.predict(x_patch)
            ScoreMap[i - image_r, j - image_r] = preds[0, 0]

            if preds[0, 0] > preds[0, 1]:
                FusionMap[i-image_r, j-image_r] = 1
                FusionResult[i-image_r, j-image_r] = img1[i-image_r, j-image_r]
            else:
                FusionMap[i-image_r, j-image_r] = 0
                FusionResult[i-image_r, j-image_r] = img2[i-image_r, j-image_r]

    io.imsave('/media/data2/mhy/data/1012N2/results_Res56v2_p64_e08/L%02d-Result.png' % (ii), FusionResult)
    io.imsave('/media/data2/mhy/data/1012N2/results_Res56v2_p64_e08/L%02d-ScoreMap.png' % (ii), ScoreMap)
    io.imsave('/media/data2/mhy/data/1012N2/results_Res56v2_p64_e08/L%02d-FusionMap.png' % (ii), FusionMap)

    # io.imshow(FusionResult)
    # plt.show()
    # io.imshow(FusionMap)
    # plt.show()

