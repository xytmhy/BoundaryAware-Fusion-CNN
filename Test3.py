

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
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
    img = tf.reshape(img, [32, 32, 1])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['label'], tf.int32)

    return img, label

xs=tf.placeholder(tf.float32,[None,32*32])
ys=tf.placeholder(tf.float32,[None,2])
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

w_fc1=weight_variable([16*16*256,512])
b_fc1=bias_variable([512])
h_conv3_flat=tf.reshape(h_conv3,[-1,16*16*256])
h_fc1=tf.nn.relu(tf.matmul(h_conv3_flat,w_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

w_fc2=weight_variable([512,2])
b_fc2=bias_variable([2])
#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
y_conv=tf.matmul(h_fc1_drop,w_fc2)+b_fc2

y = tf.nn.softmax(y_conv)

#cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv))  # 定义交叉熵为loss函数
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # 调用优化器优化，其实就是通过喂数据争取cross_entropy最小化

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# img,label=read_and_decode("FusionTrain.tfrecords")
# img_batch,label_batch=tf.train.batch([img,label],batch_size=50,capacity=2000)
def pares_tf(example_proto):
    #定义解析的字典
    dics = {}
    dics['label'] = tf.FixedLenFeature(shape=[],dtype=tf.int64)
    dics['image'] = tf.FixedLenFeature(shape=[],dtype=tf.string)
    #调用接口解析一行样本
    parsed_example = tf.parse_single_example(serialized=example_proto,features=dics)
    image = tf.decode_raw(parsed_example['image'],out_type=tf.uint8)
    image = tf.reshape(image,shape=[32*32])
    #这里对图像数据做归一化，是关键，没有这句话，精度不收敛，为0.1左右，
    # 有了这里的归一化处理，精度与原始数据一致
    image = tf.cast(image,tf.float32)*(1./255)-0.5
    label = parsed_example['label']
    label = tf.cast(label,tf.int32)
    label = tf.one_hot(label, depth=2, on_value=1)
    return image,label

dataset=tf.data.TFRecordDataset(filenames=['FusionTrain.tfrecords'])
dataset=dataset.map(read_and_decode("FusionTrain.tfrecords"))
dataset=dataset.batch(50).repeat(1)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

correct_predict = tf.equal(tf.argmax(y,1),tf.argmax(ys,1))
#定义如何计算准确率
accuracy = tf.reduce_mean(tf.cast(correct_predict,dtype=tf.float32),name="accuracy")
#定义初始化op
init = tf.global_variables_initializer()

with tf.Session() as sess:
    print("start")
    sess.run(fetches=init)
    i = 0
    try:
        while True:
            #通过session每次从数据集中取值
            image,label= sess.run(fetches=next_element)
            sess.run(fetches=train_step, feed_dict={xs: image, ys: label})
            if i % 100 == 0:
                train_accuracy = sess.run(fetches=accuracy, feed_dict={xs: image, ys: label})
                print(i, "accuracy=", train_accuracy)
            i = i + 1
    except tf.errors.OutOfRangeError:
        print("end!")

