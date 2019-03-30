import tensorflow as tf
import numpy as np
import cv2
from skimage import io
from matplotlib import pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES']='1'

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

sess = tf.Session()
# step1网络结构图
saver = tf.train.import_meta_graph('/media/data2/mhy/models/model1/SimpleFusion.ckpt-20000.meta')
# step2加载权重参数
saver.restore(sess, '/media/data2/mhy/models/model1/SimpleFusion.ckpt-20000')

# coord = tf.train.Coordinator()
# threads = tf.train.start_queue_runners(coord=coord)

image_r=int(image_size/2)
for i in range(image_r,img1.shape[0]-image_r):
    for j in range(image_r,img1.shape[1]-image_r):
        patch1=img1[i-image_r:i+image_r,j-image_r:j+image_r]
        patch2=img2[i-image_r:i+image_r,j-image_r:j+image_r]

        patch=np.zeros([image_size,image_size,2])
        patch[:,:,0]=patch2
        patch[:,:,1]=patch1

        x_batch = patch.reshape(1, image_size, image_size, num_channels)

        # 获取默认的图
        graph = tf.get_default_graph()
        y_pred = graph.get_tensor_by_name("y_pred:0")
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, 2))
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = sess.run(y_pred, feed_dict_testing)

        if result[0,0]>result[0,1]:
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

        # if (result1[0,1])>(result2[0,1]) and (result1[0,1])>(result2[0,0]):
        #         FusionMap[i, j] = 0
        #         FusionResult[i,j]=img2[i,j]
        # else:
        #         if(result1[0,0])>(result2[0,0]) and (result1[0,0])>(result1[0,1]):
        #                 FusionMap[i, j] = 1
        #                 FusionResult[i,j]=img1[i,j]
        #         else:
        #                 FusionMap[i, j] = 0.5
        #                 FusionResult[i, j] = img1[i, j]


io.imshow(FusionResult)
plt.show()
io.imshow(FusionMap)
plt.show()

io.imsave('./ImgData/FusionResult.jpg',FusionResult)
io.imsave('./ImgData/FusionMap.jpg',FusionMap)





#
# direct = os.listdir(path)
# for file in direct:
#     image = cv2.imread(path + '/' + file)
#     print("adress:", path + '/' + file)
#     image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
#     images.append(image)
# images = np.array(images, dtype=np.uint8)
# images = images.astype('float32')
# images = np.multiply(images, 1.0 / 255.0)
#
# for img in images:
#     x_batch = img.reshape(1, image_size, image_size, num_channels)
#
#     sess = tf.Session()
#
#     # step1网络结构图
#     saver = tf.train.import_meta_graph('./model/SimpleFusion.ckpt-9990.meta')
#
#     # step2加载权重参数
#     saver.restore(sess, './model/SimpleFusion.ckpt-9990')
#
#     # 获取默认的图
#     graph = tf.get_default_graph()
#
#     y_pred = graph.get_tensor_by_name("y_pred:0")
#
#     x = graph.get_tensor_by_name("x:0")
#     y_true = graph.get_tensor_by_name("y_true:0")
#     y_test_images = np.zeros((1, 2))
#
#     feed_dict_testing = {x: x_batch, y_true: y_test_images}
#     result = sess.run(y_pred, feed_dict_testing)
#
#     res_label = ['p11', 'p22']
#     print(res_label[result.argmax()])