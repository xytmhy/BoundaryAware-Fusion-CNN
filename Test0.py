

import tensorflow as tf
sess=tf.InteractiveSession()

def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def read_and_decode(filename):
    #根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw' : tf.FixedLenFeature([], tf.string),
    })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [32*32])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

xs=tf.placeholder(tf.float32,[None,32*32])
ys=tf.placeholder(tf.float32,[None,1])
keep_prob=tf.placeholder(tf.float32)
x_image=tf.reshape(xs,[-1,32,32,1])


w_conv1=weight_variable([3,3,1,64])
b_conv1=bias_variable([64])
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
print(h_conv1)

w_conv2=weight_variable([3,3,64,128])
b_conv2=bias_variable([128])
h_conv2=tf.nn.relu(conv2d(h_conv1,w_conv2)+b_conv2)
print(h_conv2)
h_pool2=max_pool_2x2(h_conv2)
print(h_pool2)

w_conv3=weight_variable([3,3,128,256])
b_conv3=bias_variable([256])
h_conv3=tf.nn.relu(conv2d(h_pool2,w_conv3)+b_conv3)
print(h_conv3)

w_fc1=weight_variable([16*16*256,256])
b_fc1=bias_variable([256])
h_conv3_flat=tf.reshape(h_conv3,[-1,16*16*256])
h_fc1=tf.nn.relu(tf.matmul(h_conv3_flat,w_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

w_fc2=weight_variable([256,1])
b_fc2=bias_variable([1])
#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
y_conv=tf.matmul(h_fc1_drop,w_fc2)+b_fc2

#cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv))  # 定义交叉熵为loss函数
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(ys,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

img,label=read_and_decode("FusionTrain.tfrecords")
print(img.shape)
print(label.shape)
img_batch,label_batch=tf.train.shuffle_batch([img,label],batch_size=50,capacity=2500000,min_after_dequeue=500)


#saver = tf.train.Saver()  # defaults to saving all variables
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(30000):
        val, l = sess.run([img_batch, label_batch])
        if i % 1000 == 0:
            print(i)
            print(val.shape)
            print(l)
        #sess.run(fetches=train_step,feed_dict={xs: val, ys: l3, keep_prob: 0.5})
    #saver.save(sess, 'model.ckpt')
# 保存模型参数，注意把这里改为自己的路径


# print("test accuracy %g" % accuracy.eval(feed_dict={
#     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

print("Finish!")

