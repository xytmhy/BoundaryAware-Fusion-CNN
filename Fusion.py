import tensorflow as tf
import numpy as np
import os, cv2
from skimage import io
from matplotlib import pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

image_size = 48
num_channels = 1
# images = []

path1 = "/media/data2/mhy/LytroDataset/lytro-01-A.jpg"
path2 = "/media/data2/mhy/LytroDataset/lytro-01-B.jpg"

img1=io.imread(path1,as_gray=True)
io.imshow(img1)
plt.show()
img2=io.imread(path2,as_gray=True)
io.imshow(img2)
plt.show()

io.imsave('F1.jpg',img1)
io.imsave('F2.jpg',img1)

# img1 = np.array(img1, dtype=np.uint8)
# img1 = img1.astype('float32')
# img1 = np.multiply(img1, 1.0 / 255.0)
# img2 = np.array(img2, dtype=np.uint8)
# img2 = img2.astype('float32')
# img2 = np.multiply(img2, 1.0 / 255.0)

FusionResult=io.imread('F1.jpg',as_gray=True)
FusionResult = FusionResult.astype('float32')
FusionResult = np.multiply(FusionResult, 1.0 / 255.0)
FusionMap=io.imread('F2.jpg',as_gray=True)
FusionMap = FusionMap.astype('float32')
FusionMap = np.multiply(FusionMap, 1.0 / 255.0)





sess = tf.Session()
# step1网络结构图
saver = tf.train.import_meta_graph('/media/data2/mhy/models/model1/SimpleFusion.ckpt-200000.meta')
# step2加载权重参数
saver.restore(sess, '/media/data2/mhy/models/model1/SimpleFusion.ckpt-200000')

r_size=int(image_size/2)
for i in range(r_size,img1.shape[0]-r_size):
    for j in range(r_size,img1.shape[1]-r_size):
# for i in range(2):
#     for j in range(2):
        patch1=img1[i-r_size:i+r_size,j-r_size:j+r_size]
        patch2=img2[i-r_size:i+r_size,j-r_size:j+r_size]

        x_batch = patch1.reshape(1, image_size, image_size, num_channels)
        # 获取默认的图
        graph = tf.get_default_graph()
        y_pred = graph.get_tensor_by_name("y_pred:0")
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, 2))
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result1 = sess.run(y_pred, feed_dict_testing)

        x_batch = patch2.reshape(1, image_size, image_size, num_channels)
        # 获取默认的图
        graph = tf.get_default_graph()
        y_pred = graph.get_tensor_by_name("y_pred:0")
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, 2))
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result2 = sess.run(y_pred, feed_dict_testing)

        # FusionMap[i, j] = result1[0,0]
        # FusionResult[i,j]=result2[0,0]





        # if result1[0,0]>result1[0,1]:
        #         if result2[0,0]<result2[0,1]:
        #                 FusionMap[i, j] = 1
        #                 FusionResult[i,j]=img1[i,j]
        #         elif result1[0,0]>result2[0,0]:
        #                 FusionMap[i, j] = 1
        #                 FusionResult[i, j] = img1[i, j]
        #         else:
        #                 FusionMap[i, j] = 0
        #                 FusionResult[i, j] = img2[i, j]
        # else:
        #         if result2[0,0]>result2[0,1]:
        #                 FusionMap[i, j] = 0
        #                 FusionResult[i,j]=img2[i,j]
        #         elif result1[0,1]>result2[0,1]:
        #                 FusionMap[i, j] = 0
        #                 FusionResult[i, j] = img2[i, j]
        #         else:
        #                 FusionMap[i, j] = 1
        #                 FusionResult[i, j] = img1[i, j]


        # print(result1[0, 0])
        # print(result1[0, 1])
        #
        if result2[0,1]>result1[0,1]:
            FusionResult[i,j]=img2[i,j]
            FusionMap[i, j]=0
        else:
            FusionMap[i, j] = 1

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

io.imsave('FusionResult.jpg',FusionResult)
io.imsave('FusionMap.jpg',FusionMap)





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